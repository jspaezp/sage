//! Linear Discriminant Analysis for FDR refinement
//!
//! "What I cannot create, I do not understand" - Richard Feynman
//!
//! One of the major reasons for the creation of Sage is to develop a search
//! engine from first principles - And when I mean first principles, I mean
//! first principles - we are going to implement a basic linear algebra system
//! (complete with Gauss-Jordan elimination and eigenvector calculation) from scratch
//! to enable LDA.

mod gauss;
pub mod kde;
pub mod matrix;
mod qvalue;
mod retention_model;
use matrix::Matrix;
pub use qvalue::assign_q_values;
use rayon::prelude::*;

use crate::{database::IndexedDatabase, scoring::Percolator};

#[allow(dead_code)]
fn all_close(lhs: &[f64], rhs: &[f64], eps: f64) -> bool {
    lhs.iter()
        .zip(rhs.iter())
        .all(|(l, r)| (l - r).abs() <= eps)
}

pub fn norm(slice: &[f64]) -> f64 {
    slice.iter().fold(0.0, |acc, x| acc + x.powi(2)).sqrt()
}

pub fn mean(slice: &[f64]) -> f64 {
    slice.iter().sum::<f64>() / slice.len() as f64
}

pub fn std(slice: &[f64]) -> f64 {
    let mean = mean(slice);
    let x = slice.iter().fold(0.0, |acc, x| acc + (x - mean).powi(2));
    (x / slice.len() as f64).sqrt()
}

pub struct LinearDiscriminantAnalysis {
    eigenvector: Vec<f64>,
}

impl LinearDiscriminantAnalysis {
    pub fn train(features: &Matrix, decoy: &[bool]) -> Option<LinearDiscriminantAnalysis> {
        assert_eq!(features.rows, decoy.len());

        // Calculate class means, and overall mean
        let x_bar = features.mean();
        let mut scatter_within = Matrix::zeros(features.cols, features.cols);
        let mut scatter_between = Matrix::zeros(features.cols, features.cols);

        let mut class_means = Vec::new();

        for class in [true, false] {
            let count = decoy.iter().filter(|&label| *label == class).count();

            let class_data = (0..features.rows)
                .into_iter()
                .zip(decoy)
                .filter(|&(_, label)| *label == class)
                .flat_map(|(row, _)| features.row(row))
                .collect::<Vec<_>>();

            let mut class_data = Matrix::new(class_data, count, features.cols);
            let mut class_mean = class_data.mean();

            // Any zeroes in the covariance matrix will cause LDA to fail:
            // Attempt to rectify by making the matrices contain 1 instead
            let ln2 = (2.0f64).ln();
            for col in class_mean.iter_mut() {
                if *col == 0.0 {
                    log::trace!(
                        "- attempting to apply correction to LDA model, where class mean = 0.0"
                    );
                    *col = -1.0;
                } else if (*col - ln2).abs() <= 1E-8 {
                    log::trace!(
                        "- attempting to apply correction to LDA model, where class mean = ln(2)"
                    );
                    *col = -1.0;
                }
            }

            for row in 0..class_data.rows {
                for col in 0..class_data.cols {
                    class_data[(row, col)] -= class_mean[col];
                }
            }

            let cov = class_data.transpose().dot(&class_data);
            scatter_within += cov;

            let diff = Matrix::col_vector(
                class_mean
                    .iter()
                    .zip(x_bar.iter())
                    .map(|(x, y)| x - y)
                    .collect::<Vec<_>>(),
            );

            scatter_between += diff.dot(&diff.transpose());
            class_means.extend(class_mean);
        }

        // Use overall mean as the initial vector for power method... seems
        // unlikely to be the actual best eigenvector!
        let mut evec = gauss::Gauss::solve(scatter_within, scatter_between)
            .map(|mat| mat.power_method(&x_bar))?;

        // In some cases, power method can return eigenvector with signs flipped -
        // Make it so that Target class scores are higher than Decoy, so that
        // we can make assumptions about this for ranking
        let class_means = Matrix::new(class_means, 2, features.cols);
        let coef = class_means.dotv(&evec);
        if coef[1] < coef[0] {
            evec.iter_mut().for_each(|c| *c *= -1.0);
        }

        log::trace!("- linear model fit with eigenvector: {:?}", evec);

        Some(LinearDiscriminantAnalysis { eigenvector: evec })
    }

    pub fn score(&self, features: &Matrix) -> Vec<f64> {
        features.dotv(&self.eigenvector)
    }
}

pub fn score_psms(db: &IndexedDatabase, scores: &mut [Percolator], predict_rt: bool) -> Option<()> {
    log::trace!("fitting linear discriminant model...");

    if predict_rt {
        // Poisson probability is usually the best single feature for refining FDR.
        // Take our set of 1% FDR filtered PSMs, and use them to train a linear
        // regression model for predicting retention time
        scores.par_sort_unstable_by(|a, b| a.poisson.total_cmp(&b.poisson));
        let passing = assign_q_values(scores);

        // Training LR might fail - not enough values, or r-squared is < 0.7
        if let Some(lr) = retention_model::RetentionModel::fit(db, &scores[..passing]) {
            log::trace!("- fit retention time model, rsq = {}", lr.r2);
            let predicted_rts = lr.predict(db, scores);
            scores
                .iter_mut()
                .zip(&predicted_rts)
                .for_each(|(score, &rt)| {
                    // LR can sometimes predict crazy values - clamp predicted RT
                    let bounded = rt.max(lr.rt_min - 10.0).min(lr.rt_max + 10.0) as f32;
                    score.predicted_rt = bounded;
                    score.delta_rt = (score.rt - bounded).powi(2) / score.rt;
                });
        }
    }

    // Declare, so that we have compile time checking of matrix dimensions
    const FEATURES: usize = 15;
    let features = scores
        .into_par_iter()
        .flat_map(|perc| {
            let poisson = match (-perc.poisson).ln_1p() {
                x if x.is_finite() => x,
                _ => 3.5,
            };

            // Transform features - LDA requires that each feature is normally
            // distributed. This is not true for all of our inputs, so we log
            // transform many of them to get them closer to a gaussian distr.
            let x: [f64; FEATURES] = [
                (perc.hyperscore).ln_1p(),
                (perc.delta_hyperscore).ln_1p(),
                (perc.delta_mass as f64).ln_1p(),
                perc.isotope_error as f64,
                perc.average_ppm as f64,
                poisson,
                (perc.matched_intensity_pct as f64).ln_1p(),
                (perc.matched_peaks as f64).ln_1p(),
                (perc.longest_b as f64).ln_1p(),
                (perc.longest_y as f64).ln_1p(),
                (perc.longest_y as f64 / perc.peptide_len as f64),
                (perc.peptide_len as f64).ln_1p(),
                (perc.missed_cleavages as f64),
                (perc.rt as f64),
                (perc.delta_rt as f64).ln_1p(),
            ];
            x
        })
        .collect::<Vec<_>>();

    let decoys = scores.iter().map(|sc| sc.label == -1).collect::<Vec<_>>();
    let features = Matrix::new(features, scores.len(), FEATURES);

    let lda = LinearDiscriminantAnalysis::train(&features, &decoys)?;
    if !lda.eigenvector.iter().all(|f| f.is_finite()) {
        log::error!(
            "linear model eigenvector includes NaN: this likely indicates a bug, please report!"
        );
        for row in 0..features.rows {
            if features.row(row).any(|f| !f.is_finite()) {
                let row = features.row(row).collect::<Vec<_>>();
                log::error!("example feature vector with NaN: {:?}", row);
                break;
            }
        }
        return None;
    }
    let discriminants = lda.score(&features);

    log::trace!("- fitting non-parametric model for posterior error probabilities");
    let kde = kde::Estimator::fit(&discriminants, &decoys);

    scores
        .iter_mut()
        .zip(&discriminants)
        .for_each(|(perc, score)| {
            perc.discriminant_score = *score as f32;
            perc.posterior_error = kde.posterior_error(*score).log10() as f32;
            if perc.posterior_error.is_infinite() {
                // This is approximately the log10 of the smallest positive
                // non-zero f64
                perc.posterior_error = -324.0;
            }
        });

    Some(())
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn linear_discriminant() {
        let a = Matrix::new([1., 2., 3., 4.], 2, 2);
        let eigenvector = [0.4159736, 0.90937671];
        assert!(all_close(
            &a.power_method(&[0.54, 0.34]),
            &eigenvector,
            1E-5
        ));

        #[rustfmt::skip]
        let feats = Matrix::new(
            [
                5., 4., 3., 2., 
                4., 5., 4., 3., 
                6., 3., 4., 5., 
                1., 0., 2., 9., 
                5., 4., 4., 3., 
                2., 1., 1., 9.5, 
                1., 0., 2., 8., 
                3., 2., -2., 10.,
            ],
            8,
            4,
        );

        let lda = LinearDiscriminantAnalysis::train(
            &feats,
            &[false, false, false, true, false, true, true, true],
        )
        .expect("error training LDA");

        let mut scores = lda.score(&feats);
        let norm = norm(&scores);
        scores = scores.into_iter().map(|s| s / norm).collect();

        let expected = [
            0.49706043,
            0.48920177,
            0.48920177,
            -0.07209359,
            0.51204672,
            -0.02849527,
            -0.04924864,
            -0.06055943,
        ];

        assert!(
            all_close(&scores, &expected, 1E-8),
            "{:?} {:?}",
            scores,
            expected
        );
    }
}
