use rayon::prelude::*;
use sage_core::{
    mass::Tolerance,
    spectrum::{Precursor, RawSpectrum, Representation},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::{cmp::Ordering, path::Path};
use timsrust::core::utils::reader::Reader as TimsRustReader;
use timsrust::core::{AcquisitionType, Converter, Frame, Im, MSLevel, Mz, ScanIndex, TofIndex};
use timsrust::tdf::{Metadata, TDFPath, TDFSpectrumReader};
use timsrust::{ImConverter, MzConverter, SpectrumReader, TimsTofPath};
use timsrust_calibration::{CalibratedTof2MzConverter, RunCalibration};

pub struct TdfReader;

/// Name of the env var that toggles which TOF-index -> m/z converter is
/// applied to fragment (MS2) and MS1 peak m/z when reading Bruker `.d`
/// (TDF) data. Set to `physical` (or `1`) to use the physical calibration
/// built from the run's own `analysis.tdf` calibration tables
/// (`timsrust-calibration`); unset or any other value uses the stock
/// (uncalibrated, sqrt-linear) timsrust converter.
///
/// Precursor m/z is a stored Bruker value (not TOF-derived) and is never
/// affected by this toggle. Ion mobility uses the stock converter in both
/// modes.
const CALIBRATION_ENV_VAR: &str = "SAGE_TDF_CALIBRATION";

fn physical_calibration_requested() -> bool {
    match std::env::var(CALIBRATION_ENV_VAR) {
        Ok(v) => {
            let v = v.trim();
            v.eq_ignore_ascii_case("physical") || v == "1"
        }
        Err(_) => false,
    }
}

/// The TOF-index -> m/z converter selected for a given file, per
/// [`CALIBRATION_ENV_VAR`]. Wraps either the stock timsrust converter or
/// the physical-calibration converter from `timsrust-calibration`.
#[derive(Clone)]
enum ChosenMzConverter {
    Stock(MzConverter),
    Calibrated(CalibratedTof2MzConverter),
}

impl Converter<TofIndex, Mz> for ChosenMzConverter {
    fn convert(&self, value: TofIndex) -> Mz {
        match self {
            ChosenMzConverter::Stock(c) => c.convert(value),
            ChosenMzConverter::Calibrated(c) => c.convert(value),
        }
    }
}

/// Read the env var once (per file parse) and build the corresponding
/// fragment/MS1 m/z converter. Logs which mode is active at INFO.
fn select_mz_converter(path: &TimsTofPath) -> ChosenMzConverter {
    if physical_calibration_requested() {
        let tdf_path = TDFPath::new(path).expect(
            "SAGE_TDF_CALIBRATION=physical requires a genuine Bruker TDF (.d) dataset",
        );
        let analysis_tdf = tdf_path.tdf().as_path().expect(
            "analysis.tdf must resolve to a local filesystem path for physical calibration",
        );
        let run_calibration = RunCalibration::from_path(analysis_tdf.to_string_lossy())
            .expect("failed to read physical calibration tables from analysis.tdf");
        let converter = run_calibration
            .mz_converter_median()
            .expect("failed to build physical (timsrust-calibration) m/z converter");
        log::info!(
            "{}=physical: using timsrust-calibration physical m/z converter for {}",
            CALIBRATION_ENV_VAR,
            path.as_ref()
        );
        ChosenMzConverter::Calibrated(converter)
    } else {
        let converter = path
            .mz_converter()
            .expect("no m/z calibration for this timsTOF dataset");
        log::info!(
            "using stock (uncalibrated) timsrust m/z converter for {}",
            path.as_ref()
        );
        ChosenMzConverter::Stock(converter)
    }
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

/// Plain, serializable mirror of [`timsrust::tdf::SpectrumProcessingParams`].
///
/// In timsrust 0.4.2, `SpectrumReaderConfig` was a concrete, directly
/// serializable struct. As of 0.5.x it is generic over the IM converter
/// (`SpectrumReaderConfig<ImC>`) and the upstream `SpectrumProcessingParams`
/// itself carries no `serde` derives. This local copy holds only the plain
/// parameters a user can actually configure, and is converted into the
/// upstream type at `parse()` time (see `BrukerMS2ProcessingConfig::into_timsrust`).
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

/// Plain, serializable mirror of [`timsrust::tdf::QuadWindowExpansionStrategy`].
///
/// The upstream enum's `UniformMobility` variant carries an
/// `Option<Arc<ImC>>` payload (the scan->IM converter used to sub-split
/// windows in mobility space), which is neither serializable nor known until
/// a dataset is opened. We drop that payload here and always pass `None` when
/// converting back to the upstream type; timsrust fills it in automatically
/// from the dataset's own (uncalibrated, for Stage 1) IM converter inside
/// `TDFSpectrumReader::new`/`FrameWindowSplittingConfiguration::finalize`.
#[derive(Deserialize, Serialize, Debug, Clone, Copy)]
pub enum BrukerQuadWindowExpansionStrategy {
    None,
    Even(usize),
    UniformMobility(f64, f64),
    UniformScan(usize, usize),
}

impl Default for BrukerQuadWindowExpansionStrategy {
    fn default() -> Self {
        Self::Even(1)
    }
}

/// Plain, serializable mirror of [`timsrust::tdf::FrameWindowSplittingConfiguration`].
#[derive(Deserialize, Serialize, Debug, Clone, Copy)]
pub enum BrukerFrameSplittingConfig {
    Quadrupole(BrukerQuadWindowExpansionStrategy),
    Window(BrukerQuadWindowExpansionStrategy),
}

impl Default for BrukerFrameSplittingConfig {
    fn default() -> Self {
        Self::Quadrupole(BrukerQuadWindowExpansionStrategy::default())
    }
}

fn convert_expansion_strategy(
    s: BrukerQuadWindowExpansionStrategy,
) -> timsrust::tdf::QuadWindowExpansionStrategy<ImConverter> {
    use timsrust::tdf::QuadWindowExpansionStrategy as Q;
    match s {
        BrukerQuadWindowExpansionStrategy::None => Q::None,
        BrukerQuadWindowExpansionStrategy::Even(n) => Q::Even(n),
        BrukerQuadWindowExpansionStrategy::UniformMobility(span, step) => {
            Q::UniformMobility((span, step), None)
        }
        BrukerQuadWindowExpansionStrategy::UniformScan(span, step) => Q::UniformScan((span, step)),
    }
}

impl From<BrukerFrameSplittingConfig>
    for timsrust::tdf::FrameWindowSplittingConfiguration<ImConverter>
{
    fn from(config: BrukerFrameSplittingConfig) -> Self {
        use timsrust::tdf::FrameWindowSplittingConfiguration as F;
        match config {
            BrukerFrameSplittingConfig::Quadrupole(s) => {
                F::Quadrupole(convert_expansion_strategy(s))
            }
            BrukerFrameSplittingConfig::Window(s) => F::Window(convert_expansion_strategy(s)),
        }
    }
}

/// Serializable mirror of `timsrust::tdf::SpectrumReaderConfig<ImConverter>`
/// (the MS2/DDA spectrum reader configuration).
#[derive(Deserialize, Serialize, Debug, Clone, Copy, Default)]
pub struct BrukerMS2ProcessingConfig {
    pub spectrum_processing_params: BrukerSpectrumProcessingParams,
    pub frame_splitting_params: BrukerFrameSplittingConfig,
}

impl BrukerMS2ProcessingConfig {
    fn into_timsrust(self) -> timsrust::tdf::SpectrumReaderConfig<ImConverter> {
        timsrust::tdf::SpectrumReaderConfig {
            spectrum_processing_params: self.spectrum_processing_params.into(),
            frame_splitting_params: self.frame_splitting_params.into(),
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

        // Read the env-var calibration toggle once per file; reused for
        // both the MS2 (fragment) and MS1 peak m/z below. Precursor m/z is
        // untouched (stored Bruker value).
        let mz_converter = select_mz_converter(&path);

        let mut spectra = match Self::open_raw_tof_spectrum_reader(&path, &path_str, &config)? {
            Some(spectrum_reader) => self.read_msn_spectra(file_id, &spectrum_reader, &mz_converter)?,
            None => {
                if physical_calibration_requested() {
                    log::warn!(
                        "{}=physical requested but '{}' is not a plain Bruker DDA-TDF dataset \
                         (MiniTdf/TSF/Parquet, or DIA-PASEF); falling back to the stock \
                         (uncalibrated) reader for MS2 fragment m/z in this file",
                        CALIBRATION_ENV_VAR,
                        path_str
                    );
                }
                let spectrum_reader = SpectrumReader::build()
                    .with_path(&path)
                    .with_config(config.ms2.into_timsrust())
                    .finalize()?;
                self.read_msn_spectra_stock(file_id, &spectrum_reader)?
            }
        };
        if requires_ms1 {
            let ms1s = self.read_ms1_spectra(&path, file_id, config.ms1, &mz_converter)?;
            spectra.extend(ms1s);
        }

        Ok(spectra)
    }

    /// Build a raw (`Spectrum<TofIndex>`) MS2 reader directly from the
    /// tdf-level API, so that fragment m/z can be computed with whichever
    /// converter [`select_mz_converter`] picked, bypassing the facade's
    /// built-in (stock, uncalibrated) conversion.
    ///
    /// Returns `Ok(None)` when `path` isn't a plain Bruker DDA-TDF dataset
    /// (MiniTdf/TSF/Parquet, or a DIA-PASEF `.d` — which the facade routes
    /// through its own centroider, not `TDFSpectrumReader`); callers should
    /// fall back to the stock facade reader in that case.
    fn open_raw_tof_spectrum_reader(
        path: &TimsTofPath,
        path_str: &str,
        config: &BrukerProcessingConfig,
    ) -> Result<Option<TDFSpectrumReader<ImConverter>>, timsrust::TimsRustError> {
        let tdf_path = match TDFPath::new(path_str) {
            Ok(p) => p,
            Err(_) => return Ok(None),
        };
        let is_dia = Metadata::new(&tdf_path)
            .map(|m| m.acquisition_type() == AcquisitionType::DIAPASEF)
            .unwrap_or(false);
        if is_dia {
            return Ok(None);
        }

        let im_converter = Arc::new(
            path.im_converter()
                .expect("no IM calibration for this timsTOF dataset"),
        );
        let reader = TDFSpectrumReader::build()
            .with_path(&tdf_path)
            .with_config(config.ms2.into_timsrust())
            .with_im_converter(im_converter)
            .finalize()
            .map_err(timsrust::SpectrumReaderError::from)?;
        Ok(Some(reader))
    }

    fn read_ms1_spectra(
        &self,
        path: &TimsTofPath,
        file_id: usize,
        config: BrukerMS1CentoidingConfig,
        mz_converter: &ChosenMzConverter,
    ) -> Result<Vec<RawSpectrum>, timsrust::TimsRustError> {
        let start = std::time::Instant::now();
        let frame_reader = timsrust::tdf::TdfFrameReader::new(path)?;
        let ims_converter = path
            .im_converter()
            .expect("no IM calibration for this timsTOF dataset");
        let tol_ppm = config.mz_ppm;
        let im_tol_pct = config.ims_pct;

        let indices: Vec<usize> = frame_reader.iter_indices().collect();
        let ms1_spectra: Vec<RawSpectrum> = indices
            .into_par_iter()
            // Filter on ms_level using the cheap, ion-less partial frame
            // before paying the cost of decompressing the full frame ions
            // (mirrors the old `FrameReader::parallel_filter` behavior).
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
                        buffer.with_frame(&frame, &ims_converter, mz_converter);

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

    /// MS2 path used when `path` is a plain Bruker DDA-TDF dataset: reads
    /// raw `Spectrum<TofIndex>` directly from the tdf-level reader and
    /// converts fragment m/z with the env-selected `mz_converter` (stock or
    /// physically calibrated). Precursor m/z is untouched.
    fn read_msn_spectra(
        &self,
        file_id: usize,
        spectrum_reader: &TDFSpectrumReader<ImConverter>,
        mz_converter: &ChosenMzConverter,
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
                                .mz_values(mz_converter)
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

    /// Fallback MS2 path (Stage 1 behavior, unchanged): used for
    /// MiniTdf/TSF/Parquet datasets and DIA-PASEF `.d` files, none of which
    /// go through [`Self::open_raw_tof_spectrum_reader`]. Always uses the
    /// facade's stock (uncalibrated) m/z conversion; the calibration
    /// toggle has no effect here.
    fn read_msn_spectra_stock(
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

    /// Generic over the converters (rather than the concrete `MzConverter`/
    /// `ImConverter` enums) so that a future calibrated converter can be
    /// dropped in with no further refactor here.
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

                let scan_index = ScanIndex::try_from(i as u32).expect("scan index out of bounds");
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
