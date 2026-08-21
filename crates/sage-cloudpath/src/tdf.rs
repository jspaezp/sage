use rayon::prelude::*;
use sage_core::{
    mass::Tolerance,
    spectrum::{Precursor, RawSpectrum, Representation},
};
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, path::Path};
use timsrust::core::AcquisitionType;
use timsrust::core::{Converter, Frame, Im, MSLevel, Mz, ScanIndex, TofIndex};
use timsrust::tdf::{Metadata, TDFPath};
use timsrust::{ImConverter, MzConverter, SpectrumReader, TimsTofPath};

pub struct TdfReader;

/// The dataset's stock TOF->m/z converter.
fn require_mz_converter(
    path: &TimsTofPath,
) -> Result<MzConverter, timsrust::tdf::MetadataReaderError> {
    path.mz_converter().ok_or_else(|| {
        timsrust::tdf::MetadataReaderError::KeyNotFound(
            "m/z calibration for this timsTOF dataset".to_string(),
        )
    })
}

/// The dataset's stock scan->1/K0 converter.
fn require_im_converter(
    path: &TimsTofPath,
) -> Result<ImConverter, timsrust::tdf::MetadataReaderError> {
    path.im_converter().ok_or_else(|| {
        timsrust::tdf::MetadataReaderError::KeyNotFound(
            "ion mobility calibration for this timsTOF dataset".to_string(),
        )
    })
}

/// True only for a Bruker DIA-PASEF `.d`; false for non-TDF formats and on any
/// metadata error.
fn is_diapasef(path_str: &str) -> bool {
    let Ok(tdf_path) = TDFPath::new(path_str) else {
        return false;
    };
    Metadata::new(&tdf_path)
        .map(|m| m.acquisition_type() == AcquisitionType::DIAPASEF)
        .unwrap_or(false)
}

#[derive(Deserialize, Serialize, Debug, Clone, Copy)]
pub struct BrukerMS1CentoidingConfig {
    pub mz_ppm: f32,
    pub ims_pct: f32,
}

impl Default for BrukerMS1CentoidingConfig {
    fn default() -> Self {
        BrukerMS1CentoidingConfig {
            mz_ppm: 5.0,
            ims_pct: 3.0,
        }
    }
}

/// Serializable mirror of [`timsrust::tdf::SpectrumProcessingParams`], which
/// carries no `serde` derives as of timsrust 0.5.
#[derive(Deserialize, Serialize, Debug, Clone, Copy)]
pub struct BrukerSpectrumProcessingParams {
    pub smoothing_window: u32,
    pub centroiding_window: u32,
    pub calibration_tolerance: f64,
    pub calibrate: bool,
}

impl Default for BrukerSpectrumProcessingParams {
    fn default() -> Self {
        let defaults = timsrust::tdf::SpectrumProcessingParams::default();
        Self {
            smoothing_window: defaults.smoothing_window,
            centroiding_window: defaults.centroiding_window,
            calibration_tolerance: defaults.calibration_tolerance,
            calibrate: defaults.calibrate,
        }
    }
}

impl From<BrukerSpectrumProcessingParams> for timsrust::tdf::SpectrumProcessingParams {
    fn from(p: BrukerSpectrumProcessingParams) -> Self {
        timsrust::tdf::SpectrumProcessingParams {
            smoothing_window: p.smoothing_window,
            centroiding_window: p.centroiding_window,
            calibration_tolerance: p.calibration_tolerance,
            calibrate: p.calibrate,
        }
    }
}

/// DDA spectrum reader configuration. Frame splitting is not exposed: DDA does
/// not use it and DIA-PASEF goes through timsrust's centroider.
#[derive(Deserialize, Serialize, Debug, Clone, Copy, Default)]
pub struct BrukerMS2ProcessingConfig {
    pub spectrum_processing_params: BrukerSpectrumProcessingParams,
}

impl BrukerMS2ProcessingConfig {
    fn into_timsrust(self) -> timsrust::tdf::SpectrumReaderConfig<ImConverter> {
        timsrust::tdf::SpectrumReaderConfig {
            spectrum_processing_params: self.spectrum_processing_params.into(),
            frame_splitting_params: Default::default(),
        }
    }
}

#[derive(Default, Deserialize, Serialize, Debug, Clone, Copy)]
pub struct BrukerProcessingConfig {
    pub ms2: BrukerMS2ProcessingConfig,
    pub ms1: BrukerMS1CentoidingConfig,
}

impl TdfReader {
    pub fn parse(
        &self,
        path_name: impl AsRef<Path>,
        file_id: usize,
        config: BrukerProcessingConfig,
        requires_ms1: bool,
    ) -> Result<Vec<RawSpectrum>, timsrust::TimsRustError> {
        let path_str = path_name.as_ref().to_string_lossy().into_owned();
        let path = TimsTofPath::new(&path_str)?;

        if is_diapasef(&path_str) {
            log::warn!(
                "{path_str}: DIA-PASEF read — timsrust centroider (precursor detection); \
                 stock (uncalibrated) converters; MS2 processing params ignored"
            );
            let mut spectra = self.read_dia_ms2_timsrust(&path, file_id)?;
            if requires_ms1 {
                spectra.extend(self.read_ms1_spectra(&path, file_id, config.ms1)?);
            }
            return Ok(spectra);
        }

        let spectrum_reader = SpectrumReader::build()
            .with_path(&path)
            .with_config(config.ms2.into_timsrust())
            .finalize()?;
        let mut spectra = self.read_msn_spectra(file_id, &spectrum_reader)?;
        if requires_ms1 {
            let ms1s = self.read_ms1_spectra(&path, file_id, config.ms1)?;
            spectra.extend(ms1s);
        }

        Ok(spectra)
    }

    fn read_ms1_spectra(
        &self,
        path: &TimsTofPath,
        file_id: usize,
        config: BrukerMS1CentoidingConfig,
    ) -> Result<Vec<RawSpectrum>, timsrust::TimsRustError> {
        let start = std::time::Instant::now();
        let frame_reader = timsrust::tdf::TdfFrameReader::new(path)?;
        let mz_converter = require_mz_converter(path)?;
        let ims_converter = require_im_converter(path)?;
        let tol_ppm = config.mz_ppm;
        let im_tol_pct = config.ims_pct;

        let indices: Vec<usize> = frame_reader.iter_indices().collect();
        let ms1_spectra: Vec<RawSpectrum> = indices
            .into_par_iter()
            // Check ms_level on the ion-less partial frame, before paying to
            // decompress the frame's ions.
            .filter_map(
                |index| match frame_reader.get_partial_frame_without_ions(index) {
                    Ok(partial_frame) => {
                        if partial_frame.info().ms_level() == MSLevel::MS1 {
                            Some(frame_reader.get_frame(index))
                        } else {
                            None
                        }
                    }
                    Err(e) => Some(Err(e)),
                },
            )
            .map_init(
                || PeakBuffer::with_capacity(2 * MAX_PEAKS),
                |buffer, frame| match frame {
                    Ok(frame) => {
                        buffer.clear();
                        buffer.with_frame(&frame, &ims_converter, &mz_converter);

                        // Squash the mobility dimension
                        let (mz, (intensity, mobility)): (Vec<f32>, (Vec<f32>, Vec<f32>)) =
                            buffer.fastcentroid_frame(tol_ppm, im_tol_pct);

                        let scan_start_time = frame.info().rt_in_seconds() as f32 / 60.0;
                        let ion_injection_time = 100.0; // This is made up, in theory we can read
                                                        // if from the tdf file
                        let total_ion_current = intensity.iter().sum::<f32>();
                        let id = frame.info().index().to_string();

                        let spec = RawSpectrum {
                            file_id,
                            precursors: vec![],
                            representation: Representation::Centroid,
                            scan_start_time,
                            ion_injection_time,
                            mz,
                            ms_level: 1,
                            id,
                            intensity,
                            total_ion_current,
                            mobility: Some(mobility),
                        };
                        Some(spec)
                    }
                    Err(x) => {
                        log::error!("error parsing spectrum: {:?}", x);
                        None
                    }
                },
            )
            .flatten()
            .collect();
        log::info!(
            "read {} ms1 spectra in {:#?}",
            ms1_spectra.len(),
            start.elapsed()
        );
        Ok(ms1_spectra)
    }

    fn read_msn_spectra(
        &self,
        file_id: usize,
        spectrum_reader: &SpectrumReader,
    ) -> Result<Vec<RawSpectrum>, timsrust::TimsRustError> {
        let spectra: Vec<RawSpectrum> = (0..spectrum_reader.len())
            .into_par_iter()
            .filter_map(|index| match spectrum_reader.get(index) {
                Ok(dda_spectrum) => match dda_spectrum.precursor() {
                    Some(dda_precursor) => {
                        let mut precursor = Self::parse_precursor(dda_precursor);
                        let width = f64::from(dda_spectrum.isolation_window().width()) as f32;
                        precursor.isolation_window =
                            Option::from(Tolerance::Da(-width / 2.0, width / 2.0));
                        let spectrum: RawSpectrum = RawSpectrum {
                            file_id,
                            precursors: vec![precursor],
                            representation: Representation::Centroid,
                            scan_start_time: f64::from(dda_precursor.rt()) as f32 / 60.0,
                            ion_injection_time: f64::from(dda_precursor.rt()) as f32,
                            total_ion_current: 0.0,
                            mz: dda_spectrum
                                .mz_values()
                                .iter()
                                .map(|&x| f64::from(x) as f32)
                                .collect(),
                            ms_level: 2,
                            id: dda_spectrum.index().to_string(),
                            intensity: dda_spectrum
                                .intensities()
                                .iter()
                                .map(|&x| x as f32)
                                .collect(),
                            mobility: None,
                        };
                        Some(spectrum)
                    }
                    None => None,
                },
                Err(_) => None,
            })
            .collect();
        Ok(spectra)
    }

    fn parse_precursor(dda_precursor: &timsrust::core::Precursor) -> Precursor {
        let mut precursor: Precursor = Precursor::default();
        precursor.mz = f64::from(dda_precursor.mz()) as f32;
        precursor.charge = dda_precursor.charge().map(|x| i8::from(x) as u8);
        precursor.intensity = dda_precursor.intensity().map(|x| x as f32);
        precursor.spectrum_ref = Option::from(dda_precursor.frame_index().to_string());
        precursor.inverse_ion_mobility = Option::from(f64::from(dda_precursor.im()) as f32);
        precursor
    }

    /// DIA-PASEF MS2 via timsrust's centroider: it deisotopes each MS1 frame to
    /// find precursors, then extracts a mobility-narrow MS2 spectrum per
    /// detected precursor.
    fn read_dia_ms2_timsrust(
        &self,
        path: &TimsTofPath,
        file_id: usize,
    ) -> Result<Vec<RawSpectrum>, timsrust::TimsRustError> {
        use timsrust::core::utils::reader::ParIterableReader as _;

        let frame_reader = timsrust::tdf::TdfFrameReader::new(path)?.into_inner();
        let mz_converter = require_mz_converter(path)?;
        let ims_converter = require_im_converter(path)?;
        // Args: min_ms1_ion_count, min_ms2_ion_count, min_spectrum_size, use_precursors.
        let reader = timsrust::centroid::spectrum_reader::SpectrumReader::new(
            frame_reader,
            0.5,
            2.0,
            5,
            true,
            ims_converter,
            mz_converter.clone(),
        )
        .map_err(|e| {
            timsrust::tdf::FrameReaderError::FileNotFound(format!(
                "timsrust DIA centroider init failed: {e}"
            ))
        })?;

        let spectra: Vec<RawSpectrum> = reader
            .par_iter()
            .filter_map(|res| {
                let spectrum = res.ok()?;
                let dda_precursor = spectrum.precursor().as_ref()?;
                let mut precursor = Self::parse_precursor(dda_precursor);
                let width = f64::from(spectrum.isolation_window().width()) as f32;
                precursor.isolation_window = Some(Tolerance::Da(-width / 2.0, width / 2.0));
                let intensity: Vec<f32> =
                    spectrum.intensities().iter().map(|&x| x as f32).collect();
                let total_ion_current: f32 = intensity.iter().sum();
                Some(RawSpectrum {
                    file_id,
                    precursors: vec![precursor],
                    representation: Representation::Centroid,
                    scan_start_time: f64::from(dda_precursor.rt()) as f32 / 60.0,
                    ion_injection_time: f64::from(dda_precursor.rt()) as f32,
                    total_ion_current,
                    mz: spectrum
                        .mz_values(mz_converter.clone())
                        .iter()
                        .map(|&x| f64::from(x) as f32)
                        .collect(),
                    ms_level: 2,
                    id: spectrum.index().to_string(),
                    intensity,
                    mobility: None,
                })
            })
            .collect();
        Ok(spectra)
    }
}

#[derive(Clone, Copy)]
struct ImsPeak {
    mz: f32,
    intensity: f32,
    im: f32,
}
const MAX_PEAKS: usize = 10_000;

/// Buffer that gets re-used on each thread to store the intermediates
/// of the centroiding for a single frame.
#[derive(Clone)]
struct PeakBuffer {
    peaks: Vec<ImsPeak>,
    order: Vec<usize>,
    agg_buff: Vec<ImsPeak>,
}

impl PeakBuffer {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            peaks: Vec::with_capacity(capacity),
            order: Vec::with_capacity(capacity),
            agg_buff: Vec::with_capacity(MAX_PEAKS),
        }
    }

    /// Generic over the converters so a calibrated converter can drop in.
    fn with_frame<M: Converter<TofIndex, Mz>, I: Converter<ScanIndex, Im>>(
        &mut self,
        frame: &Frame,
        ims_converter: &I,
        mz_converter: &M,
    ) {
        let tof_indices = frame.ions().tof_indices();
        let intensities = frame.ions().intensities();
        let scan_offsets = frame.ions().scan_offsets();

        let expect_len = tof_indices.len();
        self.expand_to_capacity(expect_len);

        let mz_iter = tof_indices
            .iter()
            .map(|&x| f64::from(mz_converter.convert(x)) as f32);
        let intensities_iter = intensities.iter().map(|&x| u32::from(x) as f32);
        let imss_iter = Self::expand_mobility_iter(scan_offsets, ims_converter);

        let peak_iter = mz_iter
            .zip(intensities_iter)
            .zip(imss_iter)
            .map(|((mz, intensity), im)| ImsPeak { mz, intensity, im });
        self.peaks.extend(peak_iter);
        assert_eq!(self.peaks.len(), expect_len);

        // sort by mz ... bc binary searching on the mz space
        // for neighbors is the fastest way to find neighbors that I have tried.
        self.peaks.sort_by(|a, b| a.mz.partial_cmp(&b.mz).unwrap());

        // The "order" is sorted by intensity
        // This will be used later during the centroiding (for details check that implementation)
        self.order.extend(0..self.len());
        self.order.sort_unstable_by(|&a, &b| {
            self.peaks[b]
                .intensity
                .partial_cmp(&self.peaks[a].intensity)
                .unwrap_or(Ordering::Equal)
        });
    }

    fn clear(&mut self) {
        self.peaks.clear();
        self.order.clear();
        self.agg_buff.clear();
    }

    fn expand_to_capacity(&mut self, capacity: usize) {
        if capacity <= self.len() {
            return;
        }
        let diff = capacity - self.len();
        // Grow by whatever is the largest 20% of the current capacity
        // or the difference.
        let diff = diff.max(self.len() / 5);

        self.peaks.reserve(diff);
        self.order.reserve(diff);
        self.agg_buff.reserve(capacity);
    }

    fn len(&self) -> usize {
        self.peaks.len()
    }

    /// Expand the scan offset slice to mobilities.
    ///
    /// The scan offsets is in essence a run-length
    /// encoded vector of scan numbers that can be converter to the 1/k0
    /// values.
    ///
    /// Essentially ... the slice [0,4,5,5], would expand to
    /// [0,0,0,0,1]; 0 to 4 have index 0, 4 to 5 have index 1, 5 to 5 would
    /// have index 2 but its empty!
    ///
    /// Then this index can be converted using the ims_converter.convert
    ///
    /// ... This should problably be implemented and exposed in timsrust.
    fn expand_mobility_iter<'a, I: Converter<ScanIndex, Im>>(
        scan_offsets: &'a [usize],
        ims_converter: &'a I,
    ) -> impl Iterator<Item = f32> + 'a {
        let ims_iter = scan_offsets
            .windows(2)
            .enumerate()
            .filter_map(|(i, w)| {
                let num = w[1] - w[0];
                if num == 0 {
                    return None;
                }
                let lo = w[0];
                let hi = w[1];

                let Ok(scan_index) = ScanIndex::try_from(i as u32) else {
                    return None;
                };
                let im = f64::from(ims_converter.convert(scan_index)) as f32;
                Some((im, lo, hi))
            })
            .flat_map(|(im, lo, hi)| (lo..hi).map(move |_| im));
        ims_iter
    }

    /// Centroiding of the IM-containing spectra
    ///
    /// This is a very rudimentary centroiding algorithm but... it seems to work well.
    /// It iterativelty goes over the peaks in decreasing intensity order and
    /// accumulates the intensity of the peaks surrounding the peak. (sort of
    /// like the first pass in dbscan).
    ///
    /// The preserved mobility and mz are the ones from the apex peak.
    /// A more complex version where the weighted mean is preserved is possible
    /// but I have seen only marginal gains and a lot more complexity + time.
    ///
    /// This dramatically reduces the number of peaks in the spectra
    /// which saves a ton of memory and time when doing LFQ, since we
    /// iterate over each peak.
    fn fastcentroid_frame(
        &mut self,
        mz_tol_ppm: f32,
        im_tol_pct: f32,
    ) -> (Vec<f32>, (Vec<f32>, Vec<f32>)) {
        // Make sure the array is mz sorted ... I should delete
        // this assertions once I am confident of the implementation.
        // but tbh, its not that slow and its simple.
        assert!(
            self.peaks.windows(2).all(|x| x[0].mz <= x[1].mz),
            "mz_array is not sorted"
        );
        assert!(self.agg_buff.is_empty(), "agg_buff is not empty");

        let mut global_num_included = 0;

        let utol = mz_tol_ppm / 1e6;
        let im_tol = im_tol_pct / 100.0;

        for &idx in &self.order {
            if self.peaks[idx].intensity <= 0.0 {
                continue;
            }
            if self.agg_buff.len() > MAX_PEAKS {
                let curr_loc_int = self.peaks[idx].intensity;
                if curr_loc_int > 200.0 {
                    log::debug!(
                        "Reached limit of the agg buffer at index {}/{} curr int={}",
                        idx,
                        self.len(),
                        curr_loc_int
                    );
                }
                break;
            }

            let mz = self.peaks[idx].mz;
            let im = self.peaks[idx].im;
            let da_tol = mz * utol;
            let left_e = mz - da_tol;
            let right_e = mz + da_tol;

            let ss_start = self.peaks.partition_point(|&x| x.mz < left_e);
            let ss_end = self.peaks.partition_point(|&x| x.mz <= right_e);

            let abs_im_tol = im * im_tol;
            let left_im = im - abs_im_tol;
            let right_im = im + abs_im_tol;

            let mut curr_intensity = 0.0;

            let mut num_includable = 0;
            for i in ss_start..ss_end {
                let im_i = self.peaks[i].im;
                if (self.peaks[i].intensity > 0.0) && im_i >= left_im && im_i <= right_im {
                    curr_intensity += self.peaks[i].intensity;
                    self.peaks[i].intensity = -1.0;
                    num_includable += 1;
                }
            }

            assert!(num_includable > 0, "At least 'itself' should be included");

            self.agg_buff.push(ImsPeak {
                mz,
                intensity: curr_intensity,
                im,
            });
            global_num_included += num_includable;

            if global_num_included == self.len() {
                log::debug!("All peaks were included in the centroiding");
                break;
            }
        }

        self.agg_buff
            .sort_unstable_by(|a, b| a.mz.partial_cmp(&b.mz).unwrap());
        // println!("Centroiding: Start len: {}; end len: {};", arr_len, result.len());
        // Ultra data is usually start: 40k end 10k,
        // HT2 data is usually start 400k end 40k, limiting to 10k
        // rarely leaves peaks with intensity > 200 ... ive never seen
        // it happen. -JSP 2025-Jan

        self.agg_buff
            .drain(..)
            .map(|x| (x.mz, (x.intensity, x.im)))
            .unzip()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Configs written against the old `frame_splitting_params` knob must still load.
    #[test]
    fn ms2_config_ignores_legacy_frame_splitting_key() {
        let cfg: BrukerMS2ProcessingConfig = serde_json::from_str(
            r#"{"frame_splitting_params": {"Quadrupole": {"Even": 3}},
                "spectrum_processing_params": {"smoothing_window": 3, "centroiding_window": 1,
                                               "calibration_tolerance": 0.1, "calibrate": false}}"#,
        )
        .unwrap();
        assert_eq!(cfg.spectrum_processing_params.smoothing_window, 3);
    }
}
