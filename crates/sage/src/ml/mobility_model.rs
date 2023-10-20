//! Retention time prediction using linear regression
//!
//! See Klammer et al., Anal. Chem. 2007, 79, 16, 6111–6118
//! https://doi.org/10.1021/ac070262k

use super::{gauss::Gauss, matrix::Matrix};
use crate::database::IndexedDatabase;
use crate::mass::VALID_AA;
use crate::peptide::Peptide;
use crate::scoring::Feature;
use rayon::prelude::*;

/// Try to fit a retention time prediction model
pub fn predict(db: &IndexedDatabase, features: &mut [Feature]) -> Option<()> {
    // Training LR might fail - not enough values, or r-squared is < 0.7
    let lr = MobilityModel::fit(db, features)?;
    features.par_iter_mut().for_each(|feat| {
        // LR can sometimes predict crazy values - clamp predicted RT
        let ims = lr.predict_peptide(db, feat);
        let bounded = ims.clamp(0.0, 2.0) as f32;
        feat.predicted_ims = bounded.sqrt();

        feat.delta_ims_model = ((feat.ims * feat.ims) - bounded).abs();
    });
    Some(())
}
pub struct MobilityModel {
    beta: Vec<f64>,
    map: [usize; 26],
    pub r2: f64,
}

const FEATURES: usize = VALID_AA.len() * 3 + 7;
const N_TERMINAL: usize = VALID_AA.len();
const C_TERMINAL: usize = VALID_AA.len() * 2;
const FIRST_R: usize = FEATURES - 7;
const FIRST_K: usize = FEATURES - 6;
const PEPTIDE_MZ: usize = FEATURES - 5;
const PEPTIDE_CHARGE: usize = FEATURES - 4;
const PEPTIDE_LEN: usize = FEATURES - 3;
const PEPTIDE_MASS: usize = FEATURES - 2;
const INTERCEPT: usize = FEATURES - 1;

// NOTE: Square mobilities are a lot more linear with respect to m/z
// than linear mobs.

// IN THEORY we could have only one model for both RT and IM
// And the RT should ignore the charge state "at training time"
impl MobilityModel {
    /// One-hot encoding of peptide sequences into feature vector
    /// Note that this currently does not take into account any modifications
    fn embed(peptide: &Peptide, charge: &u8, map: &[usize; 26]) -> [f64; FEATURES] {
        let k_idx: usize = map[(b'K' - b'A') as usize];
        let r_idx: usize = map[(b'R' - b'A') as usize];
        let mut embedding = [0.0; FEATURES];
        let cterm = peptide.sequence.len().saturating_sub(3);
        let pep_length = peptide.sequence.len() as f64;

        let default_first_val = 1.0f64;
        // let default_first_val = 0f64;
        embedding[FIRST_K] = default_first_val;
        embedding[FIRST_R] = default_first_val;

        for (aa_idx, residue) in peptide.sequence.iter().enumerate() {
            let idx = map[(residue - b'A') as usize];
            embedding[idx] += 1.0;
            // Embed N- and C-terminal AA's (2 on each end, excluding K/R)
            match aa_idx {
                0 | 1 => embedding[N_TERMINAL + idx] += 1.0,
                x if x == cterm || x == cterm + 1 => embedding[C_TERMINAL + idx] += 1.0,
                _ => {}
            }
            match idx {
                x if x == r_idx => {
                    if embedding[FIRST_R] == default_first_val {
                        // embedding[FIRST_R] = (pep_length - aa_idx as f64)/pep_length;
                        embedding[FIRST_R] = (aa_idx as f64 + 1.) / pep_length;
                    }
                }
                x if x == k_idx => {
                    if embedding[FIRST_K] == default_first_val {
                        // embedding[FIRST_K] = (pep_length - aa_idx as f64)/pep_length;
                        embedding[FIRST_K] = (aa_idx as f64 + 1.) / pep_length;
                    }
                }
                _ => {}
            }
        }
        let charge_feature: f64 = *charge as f64;
        embedding[PEPTIDE_CHARGE] = charge_feature;
        embedding[PEPTIDE_LEN] = peptide.sequence.len() as f64;
        embedding[PEPTIDE_MASS] = (peptide.monoisotopic as f64) / 1000.0;
        embedding[PEPTIDE_MZ] = ((peptide.monoisotopic as f64) / charge_feature) / 1000.0;
        embedding[INTERCEPT] = 1.0;
        embedding
    }

    /// Attempt to fit a linear regression model: peptide sequence ~ retention time
    pub fn fit(db: &IndexedDatabase, training_set: &[Feature]) -> Option<Self> {
        // Create a mapping from amino acid character to vector embedding
        // Q: Why has this been implemented as a this map and not a hashmap[byte -> usize]?
        let mut map = [0; 26];
        for (idx, aa) in VALID_AA.iter().enumerate() {
            map[(aa - b'A') as usize] = idx;
        }

        let ims = training_set
            .par_iter()
            .filter(|feat| feat.label == 1 && feat.spectrum_q <= 0.01)
            .map(|psm| psm.ims as f64)
            .collect::<Vec<f64>>();

        // NOTE: Here the model is trained on the square mobilities.
        let ims = ims.into_iter().map(|x| x * x).collect::<Vec<f64>>();

        let ims_mean = ims.iter().sum::<f64>() / ims.len() as f64;
        let ims_var = ims.iter().map(|rt| (rt - ims_mean).powi(2)).sum::<f64>();

        let rt = Matrix::col_vector(ims);

        let features = training_set
            .par_iter()
            .filter(|feat| feat.label == 1 && feat.spectrum_q <= 0.01)
            .flat_map_iter(|psm| Self::embed(&db[psm.peptide_idx], &psm.charge, &map))
            .collect::<Vec<_>>();

        let rows = features.len() / FEATURES;
        let features = Matrix::new(features, rows, FEATURES);

        let f_t = features.transpose();
        let cov = f_t.dot(&features);
        let b = f_t.dot(&rt);

        log::info!("Solving beta");
        let beta = Gauss::solve(cov, b)?;

        let predicted_rt = features.dot(&beta).take();
        let sum_squared_error = predicted_rt
            .iter()
            .zip(rt.take())
            .map(|(pred, act)| (pred - act).powi(2))
            .sum::<f64>();

        let r2 = 1.0 - (sum_squared_error / ims_var);
        Some(Self {
            beta: beta.take(),
            map,
            r2,
        })
    }

    /// Predict retention times for a collection of PSMs
    pub fn predict_peptide(&self, db: &IndexedDatabase, psm: &Feature) -> f64 {
        let v = Self::embed(&db[psm.peptide_idx], &psm.charge, &self.map);
        v.into_iter()
            .zip(&self.beta)
            .fold(0.0f64, |sum, (x, y)| sum + x * y)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::enzyme::Digest;

    #[test]
    fn test_feature_embed() {
        let peps = vec![
            Peptide::try_from(Digest {
                decoy: false,
                sequence: "LEKSLIEK".into(),
                missed_cleavages: 0,
                ..Default::default()
            })
            .unwrap(),
            Peptide::try_from(Digest {
                decoy: false,
                sequence: "LERSLIEK".into(),
                missed_cleavages: 0,
                ..Default::default()
            })
            .unwrap(),
            Peptide::try_from(Digest {
                decoy: false,
                sequence: "LESLIEK".into(),
                missed_cleavages: 0,
                ..Default::default()
            })
            .unwrap(),
        ];

        let charge = 2;
        let mut map = [0; 26];
        for (idx, aa) in VALID_AA.iter().enumerate() {
            map[(aa - b'A') as usize] = idx;
        }
        let embeddings: Vec<[f64; FEATURES]> = peps
            .iter()
            .map(|x| MobilityModel::embed(x, &charge, &map))
            .collect();

        let first_ks = embeddings.iter().map(|x| x[FIRST_K]).collect::<Vec<f64>>();
        let first_rs = embeddings.iter().map(|x| x[FIRST_R]).collect::<Vec<f64>>();
        assert_eq!(first_ks, vec![0.375, 1.0, 1.0]);
        assert_eq!(first_rs, vec![1.0, 0.375, 1.0]);
    }
}
