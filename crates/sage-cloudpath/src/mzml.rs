use async_compression::tokio::bufread::ZlibDecoder;
use quick_xml::events::{BytesEnd, BytesStart, BytesText, Event};
use quick_xml::Reader;
use sage_core::spectrum::{Precursor, Representation};
use sage_core::{mass::Tolerance, spectrum::RawSpectrum};
use std::collections::HashMap;
use tokio::io::{AsyncBufRead, AsyncReadExt};

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
/// Which tag are we inside?
enum State {
    Spectrum,
    Scan,
    BinaryDataArray,
    Binary,
    Precursor,
    SelectedIon,
}

#[derive(Copy, Clone, Debug)]
enum BinaryKind {
    Intensity,
    Mz,
    Noise,
}

#[derive(Copy, Clone, Debug)]
enum Dtype {
    F32,
    F64,
}

// MUST supply only one of the following
const ZLIB_COMPRESSION: &[u8] = b"MS:1000574";
const NO_COMPRESSION: &[u8] = b"MS:1000576";

// MUST supply only one of the following
const INTENSITY_ARRAY: &[u8] = b"MS:1000515";
const MZ_ARRAY: &[u8] = b"MS:1000514";
const NOISE_ARRAY: &[u8] = b"MS:1002744";

// MUST supply only one of the following
const FLOAT_64: &[u8] = b"MS:1000523";
const FLOAT_32: &[u8] = b"MS:1000521";

const MS_LEVEL: &[u8] = b"MS:1000511";
const PROFILE: &[u8] = b"MS:1000128";
const CENTROID: &[u8] = b"MS:1000127";
const TOTAL_ION_CURRENT: &[u8] = b"MS:1000285";

const SCAN_START_TIME: &[u8] = b"MS:1000016";
const UNIT_SECONDS: &[u8] = b"UO:0000010";
const UNIT_MINUTES: &[u8] = b"UO:0000031";
const ION_INJECTION_TIME: &[u8] = b"MS:1000927";

const SELECTED_ION_MZ: &[u8] = b"MS:1000744";
const SELECTED_ION_INT: &[u8] = b"MS:1000042";
const SELECTED_ION_CHARGE: &[u8] = b"MS:1000041";

const ISO_WINDOW_TARGET: &[u8] = b"MS:1000827";
const ISO_WINDOW_LOWER: &[u8] = b"MS:1000828";
const ISO_WINDOW_UPPER: &[u8] = b"MS:1000829";

const INVERSE_ION_MOBILITY: &[u8] = b"MS:1002815";

pub struct MzMLReader {
    ms_level: Option<u8>,
    // If set to Some(level) and noise intensities are present in the MzML file,
    // divide intensities at this MS-level by noise to calculate S/N
    signal_to_noise: Option<u8>,

    file_id: usize,
}

impl MzMLReader {
    /// Create a new [`MzMlReader`] with a minimum MS level filter
    ///
    /// # Example
    ///
    /// A minimum level of 2 will not parse or return MS1 scans
    pub fn with_file_id_and_level_filter(file_id: usize, ms_level: u8) -> Self {
        Self {
            ms_level: Some(ms_level),
            file_id,
            signal_to_noise: None,
        }
    }

    pub fn with_file_id(file_id: usize) -> Self {
        Self {
            ms_level: None,
            signal_to_noise: None,
            file_id,
        }
    }

    pub fn set_file_id(&mut self, file_id: usize) -> &mut Self {
        self.file_id = file_id;
        self
    }

    pub fn set_signal_to_noise(&mut self, sn: Option<u8>) -> &mut Self {
        self.signal_to_noise = sn;
        self
    }

    /// Here be dragons -
    /// Seriously, this kinda sucks because it's a giant imperative, stateful loop.
    /// But I also don't want to spend any more time working on an mzML parser...
    pub async fn parse<B: AsyncBufRead + Unpin>(
        &self,
        b: B,
    ) -> Result<Vec<RawSpectrum>, MzMLError> {
        let mut reader = Reader::from_reader(b);
        let mut buf = Vec::new();

        let mut output_buffer = Vec::with_capacity(4096);

        let mut parser = ParseState::new(self.file_id, self.ms_level, self.signal_to_noise);

        loop {
            match reader.read_event_into_async(&mut buf).await {
                Ok(Event::Start(ref ev)) => parser.on_start(ev)?,
                Ok(Event::Empty(ref ev)) => parser.on_empty(ev)?,
                Ok(Event::Text(text)) => parser.on_binary(&text, &mut output_buffer).await?,
                Ok(Event::End(ref ev)) => parser.on_end(ev)?,
                Ok(Event::Eof) => break,
                Ok(_) => {}
                Err(err) => {
                    log::error!("unhandled XML error while parsing mzML: {}", err)
                }
            }
            buf.clear();
        }
        Ok(parser.spectra)
    }
}

/// A required attribute, or [`MzMLError::Malformed`].
fn get_attr<'a>(ev: &'a BytesStart, key: &[u8]) -> Result<std::borrow::Cow<'a, [u8]>, MzMLError> {
    Ok(ev
        .try_get_attribute(key)?
        .ok_or(MzMLError::Malformed)?
        .value)
}

/// Parse a required attribute into `T`.
fn parse_attr<T>(ev: &BytesStart, key: &[u8]) -> Result<T, MzMLError>
where
    T: std::str::FromStr,
    MzMLError: From<<T as std::str::FromStr>::Err>,
{
    let v = get_attr(ev, key)?;
    Ok(std::str::from_utf8(&v)?.parse()?)
}

/// A cvParam resolved from its accession (the spectrum/binaryDataArray-level
/// params - all a `<referenceableParamGroup>` carries in practice).
#[derive(Copy, Clone)]
enum Param {
    MsLevel(u8),
    Centroid,
    Profile,
    TotalIonCurrent(f32),
    Compression(bool),
    Dtype(Dtype),
    ArrayKind(BinaryKind),
    Ignored,
}

/// Resolve accession + value into a [`Param`]; parsing here surfaces group value
/// errors at definition time.
///
/// Precursor/selectedIon/scan params (isolation window, charge, mobility) resolve
/// to [`Param::Ignored`]: mzML allows groups in those scopes, but no converter
/// we've seen uses them (ProteoWizard/SCIEX emit those inline).
fn resolve(acc: &[u8], value: Option<&str>) -> Result<Param, MzMLError> {
    Ok(match acc {
        MS_LEVEL => Param::MsLevel(value.ok_or(MzMLError::Malformed)?.parse()?),
        CENTROID => Param::Centroid,
        PROFILE => Param::Profile,
        TOTAL_ION_CURRENT => Param::TotalIonCurrent(value.ok_or(MzMLError::Malformed)?.parse()?),
        ZLIB_COMPRESSION => Param::Compression(true),
        NO_COMPRESSION => Param::Compression(false),
        FLOAT_64 => Param::Dtype(Dtype::F64),
        FLOAT_32 => Param::Dtype(Dtype::F32),
        INTENSITY_ARRAY => Param::ArrayKind(BinaryKind::Intensity),
        MZ_ARRAY => Param::ArrayKind(BinaryKind::Mz),
        NOISE_ARRAY => Param::ArrayKind(BinaryKind::Noise),
        _ => Param::Ignored,
    })
}

/// Resolve a cvParam event into a [`Param`].
fn resolve_param(ev: &BytesStart) -> Result<Param, MzMLError> {
    let acc = get_attr(ev, b"accession")?;
    let value = ev.try_get_attribute(b"value")?;
    let value = match &value {
        Some(a) => Some(std::str::from_utf8(&a.value)?),
        None => None,
    };
    resolve(acc.as_ref(), value)
}

/// Parser state, lifted out of the event loop into per-event methods.
///
/// Holds no reader/buffers: an [`Event`] borrows the read buffer, so keeping it
/// in the pump lets these methods take `&mut self`. Config is copied in - no
/// lifetime parameter.
struct ParseState {
    file_id: usize,
    ms_level: Option<u8>,
    signal_to_noise: Option<u8>,

    state: Option<State>,
    spectrum: RawSpectrum,
    precursor: Precursor,
    iso_window_lo: Option<f32>,
    iso_window_hi: Option<f32>,
    noise_array: Vec<f32>,
    compression: bool,
    binary_dtype: Dtype,
    binary_array: Option<BinaryKind>,

    // cvParams shared via <referenceableParamGroup> and pulled in by
    // <referenceableParamGroupRef>; resolved once, replayed through `apply`.
    ref_params: HashMap<String, Vec<Param>>,
    current_ref_group: Option<String>,

    spectra: Vec<RawSpectrum>,
}

impl ParseState {
    fn new(file_id: usize, ms_level: Option<u8>, signal_to_noise: Option<u8>) -> Self {
        Self {
            file_id,
            ms_level,
            signal_to_noise,
            state: None,
            spectrum: RawSpectrum::default_with_file_id(file_id),
            precursor: Precursor::default(),
            iso_window_lo: None,
            iso_window_hi: None,
            noise_array: Vec::new(),
            compression: false,
            binary_dtype: Dtype::F64,
            binary_array: None,
            ref_params: HashMap::new(),
            current_ref_group: None,
            spectra: Vec::new(),
        }
    }

    /// Apply a resolved cvParam under the current state - the single path shared
    /// by inline cvParams and replayed group params.
    fn apply(&mut self, param: Param) {
        match (self.state, param) {
            (Some(State::Spectrum), Param::MsLevel(level)) => {
                if let Some(filter) = self.ms_level {
                    if level != filter {
                        self.spectrum = RawSpectrum::default_with_file_id(self.file_id);
                        self.state = None;
                    }
                }
                self.spectrum.ms_level = level;
            }
            (Some(State::Spectrum), Param::Profile) => {
                self.spectrum.representation = Representation::Profile
            }
            (Some(State::Spectrum), Param::Centroid) => {
                self.spectrum.representation = Representation::Centroid
            }
            (Some(State::Spectrum), Param::TotalIonCurrent(value)) => {
                if value == 0.0 {
                    // No ion current, break out of current state
                    self.spectrum = RawSpectrum::default_with_file_id(self.file_id);
                    self.state = None;
                } else {
                    self.spectrum.total_ion_current = value;
                }
            }
            (Some(State::BinaryDataArray), Param::Compression(c)) => self.compression = c,
            (Some(State::BinaryDataArray), Param::Dtype(d)) => self.binary_dtype = d,
            (Some(State::BinaryDataArray), Param::ArrayKind(k)) => self.binary_array = Some(k),
            // Unknown CV inside a binary array - perhaps noise
            (Some(State::BinaryDataArray), _) => self.binary_array = None,
            _ => {}
        }
    }

    fn on_start(&mut self, ev: &BytesStart) -> Result<(), MzMLError> {
        // State transition into child tag
        self.state = match (ev.name().into_inner(), self.state) {
            (b"spectrum", _) => Some(State::Spectrum),
            (b"scan", Some(State::Spectrum)) => Some(State::Scan),
            (b"binaryDataArray", Some(State::Spectrum)) => Some(State::BinaryDataArray),
            (b"binary", Some(State::BinaryDataArray)) => Some(State::Binary),
            (b"precursor", Some(State::Spectrum)) => Some(State::Precursor),
            (b"selectedIon", Some(State::Precursor)) => Some(State::SelectedIon),
            _ => self.state,
        };
        match ev.name().into_inner() {
            b"spectrum" => {
                let id = get_attr(ev, b"id")?;
                self.spectrum.id = std::str::from_utf8(&id)?.to_string();
            }
            b"precursor" => {
                // Not all precursor fields have a spectrumRef
                if let Some(scan) = ev.try_get_attribute(b"spectrumRef")? {
                    let scan = std::str::from_utf8(&scan.value)?;
                    self.precursor.spectrum_ref = Some(scan.to_string())
                }
            }
            b"referenceableParamGroup" => {
                let id = get_attr(ev, b"id")?;
                let id = std::str::from_utf8(&id)?.to_string();
                self.ref_params.entry(id.clone()).or_default();
                self.current_ref_group = Some(id);
            }
            _ => {}
        }
        Ok(())
    }

    fn on_empty(&mut self, ev: &BytesStart) -> Result<(), MzMLError> {
        // Inside a group definition: stash resolved params for later replay.
        if let Some(group) = &self.current_ref_group {
            if ev.name().into_inner() == b"cvParam" {
                let param = resolve_param(ev)?;
                if let Some(params) = self.ref_params.get_mut(group) {
                    params.push(param);
                }
                return Ok(());
            }
        }
        match (self.state, ev.name().into_inner()) {
            (_, b"referenceableParamGroupRef") => {
                let id = get_attr(ev, b"ref")?;
                let id = std::str::from_utf8(&id)?.to_string();
                // Clone: release the &ref_params borrow before apply takes &mut self.
                if let Some(params) = self.ref_params.get(&id).cloned() {
                    for param in params {
                        self.apply(param);
                    }
                }
            }
            (Some(State::Spectrum), b"cvParam") | (Some(State::BinaryDataArray), b"cvParam") => {
                let param = resolve_param(ev)?;
                self.apply(param);
            }
            (Some(State::Precursor), b"cvParam") => {
                let accession = get_attr(ev, b"accession")?;
                match accession.as_ref() {
                    ISO_WINDOW_TARGET => {
                        // use isolation window target for precursor m/z, e.g. to handle
                        // DIA setups where the mzML conversion software doesn't write
                        // a selection ion tag
                        if self.precursor.mz == 0.0 {
                            self.precursor.mz = parse_attr(ev, b"value")?
                        }
                    }
                    ISO_WINDOW_LOWER => self.iso_window_lo = Some(parse_attr(ev, b"value")?),
                    ISO_WINDOW_UPPER => self.iso_window_hi = Some(parse_attr(ev, b"value")?),
                    _ => {}
                }
            }
            (Some(State::SelectedIon), b"cvParam") => {
                let accession = get_attr(ev, b"accession")?;
                match accession.as_ref() {
                    SELECTED_ION_CHARGE => {
                        self.precursor.charge = Some(parse_attr(ev, b"value")?);
                    }
                    SELECTED_ION_MZ => {
                        let val = parse_attr(ev, b"value")?;
                        if val != 0.0 {
                            self.precursor.mz = val;
                        }
                    }
                    SELECTED_ION_INT => {
                        self.precursor.intensity = Some(parse_attr(ev, b"value")?);
                    }
                    INVERSE_ION_MOBILITY => {
                        self.precursor.inverse_ion_mobility = Some(parse_attr(ev, b"value")?);
                    }
                    _ => {}
                }
            }
            (Some(State::Scan), b"cvParam") => {
                let accession = get_attr(ev, b"accession")?;
                match accession.as_ref() {
                    SCAN_START_TIME => {
                        let scan_start_time: f32 = parse_attr(ev, b"value")?;
                        let unit = get_attr(ev, b"unitAccession")?;

                        self.spectrum.scan_start_time = match unit.as_ref() {
                            UNIT_SECONDS => scan_start_time / 60.0,
                            UNIT_MINUTES => scan_start_time,
                            _ => return Err(MzMLError::Malformed),
                        };
                    }
                    ION_INJECTION_TIME => {
                        self.spectrum.ion_injection_time = parse_attr(ev, b"value")?;
                    }
                    INVERSE_ION_MOBILITY => {
                        self.precursor.inverse_ion_mobility = Some(parse_attr(ev, b"value")?);
                    }
                    _ => {}
                }
            }
            _ => {}
        }
        Ok(())
    }

    async fn on_binary(
        &mut self,
        text: &BytesText<'_>,
        output_buffer: &mut Vec<u8>,
    ) -> Result<(), MzMLError> {
        if self.state != Some(State::Binary) {
            return Ok(());
        }
        if let Some(filter) = self.ms_level {
            if self.spectrum.ms_level != filter {
                return Ok(());
            }
        }
        let raw = text.unescape()?;
        // There are occasionally empty binary data arrays, or unknown CVs
        if raw.is_empty() || self.binary_array.is_none() {
            return Ok(());
        }
        let decoded = base64::decode(raw.as_bytes())?;
        let bytes = match self.compression {
            false => &decoded,
            true => {
                let mut r = ZlibDecoder::new(decoded.as_slice());
                let n = r.read_to_end(output_buffer).await?;
                &output_buffer[..n]
            }
        };

        let array = match self.binary_dtype {
            Dtype::F32 => {
                let mut buf: [u8; 4] = [0; 4];
                bytes
                    .chunks(4)
                    .filter(|chunk| chunk.len() == 4)
                    .map(|chunk| {
                        buf.copy_from_slice(chunk);
                        f32::from_le_bytes(buf)
                    })
                    .collect::<Vec<f32>>()
            }
            Dtype::F64 => {
                let mut buf: [u8; 8] = [0; 8];
                bytes
                    .chunks(8)
                    .map(|chunk| {
                        buf.copy_from_slice(chunk);
                        f64::from_le_bytes(buf) as f32
                    })
                    .collect::<Vec<f32>>()
            }
        };
        output_buffer.clear();

        match self.binary_array {
            Some(BinaryKind::Intensity) => {
                self.spectrum.intensity = array;
            }
            Some(BinaryKind::Mz) => {
                self.spectrum.mz = array;
            }
            Some(BinaryKind::Noise) => {
                self.noise_array = array;
            }
            None => {}
        }

        self.binary_array = None;
        Ok(())
    }

    fn on_end(&mut self, ev: &BytesEnd) -> Result<(), MzMLError> {
        self.state = match (self.state, ev.name().into_inner()) {
            (Some(State::Binary), b"binary") => Some(State::BinaryDataArray),
            (Some(State::BinaryDataArray), b"binaryDataArray") => Some(State::Spectrum),
            (Some(State::SelectedIon), b"selectedIon") => Some(State::Precursor),
            (Some(State::Precursor), b"precursor") => {
                if self.precursor.mz != 0.0 {
                    self.precursor.isolation_window = match (self.iso_window_lo, self.iso_window_hi)
                    {
                        (Some(lo), Some(hi)) => Some(Tolerance::Da(-lo, hi)),
                        _ => None,
                    };
                    let precursor = std::mem::take(&mut self.precursor);
                    self.spectrum.precursors.push(precursor);
                }
                Some(State::Spectrum)
            }
            (Some(State::Scan), b"scan") => Some(State::Spectrum),
            (_, b"referenceableParamGroup") => {
                self.current_ref_group = None;
                self.state
            }
            (_, b"spectrum") => {
                let allow = self
                    .ms_level
                    .as_ref()
                    .map(|&level| level == self.spectrum.ms_level)
                    .unwrap_or(true);

                let keep = match (allow, self.signal_to_noise) {
                    (true, Some(level))
                        if level == self.spectrum.ms_level && !self.noise_array.is_empty() =>
                    {
                        self.spectrum
                            .intensity
                            .iter_mut()
                            .zip(self.noise_array.iter())
                            .for_each(|(int, noise)| *int /= noise);
                        self.noise_array.clear();
                        true
                    }
                    (true, _) => true,
                    (false, _) => false,
                };

                let spectrum = std::mem::replace(
                    &mut self.spectrum,
                    RawSpectrum::default_with_file_id(self.file_id),
                );
                if keep {
                    self.spectra.push(spectrum);
                }
                None
            }
            _ => self.state,
        };
        Ok(())
    }
}

#[derive(thiserror::Error, Debug)]
pub enum MzMLError {
    #[error("malformed MzML")]
    Malformed,
    #[error("unsupported cvParam {0}")]
    UnsupportedCV(String),
    #[error("XML parsing error: {0}")]
    XMLError(#[from] quick_xml::Error),
    #[error("io error: {0}")]
    IOError(#[from] std::io::Error),
    #[error("utf8 error: {0}")]
    Utf8Error(#[from] std::str::Utf8Error),
    #[error("error parsing float: {0}")]
    FloatError(#[from] std::num::ParseFloatError),
    #[error("error parsing int: {0}")]
    IntError(#[from] std::num::ParseIntError),
    #[error("error decoding base64: {0}")]
    Base64Error(#[from] base64::DecodeError),
}

#[cfg(test)]
mod test {
    use sage_core::{mass::Tolerance, spectrum::Representation};

    use super::{MzMLError, MzMLReader};

    #[tokio::test]
    async fn parse_spectrum_issue_78() -> Result<(), MzMLError> {
        let s = r#"
        <spectrum id="spectrum=2442" index="286" defaultArrayLength="102" dataProcessingRef="dp_sp_1">
            <cvParam cvRef="MS" accession="MS:1000127" name="centroid spectrum" />
            <cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="2" />
            <cvParam cvRef="MS" accession="MS:1000294" name="mass spectrum" />
            <cvParam cvRef="MS" accession="MS:1000130" name="positive scan" />
            <cvParam cvRef="MS" accession="MS:1000504" name="base peak m/z" value="638.352905273437955"/>
            <cvParam cvRef="MS" accession="MS:1000505" name="base peak intensity" value="113.885513305664006"/>
            <cvParam cvRef="MS" accession="MS:1000285" name="total ion current" value="793.395202636718977"/>
            <cvParam cvRef="MS" accession="MS:1000528" name="lowest observed m/z" value="147.290603637695"/>
            <cvParam cvRef="MS" accession="MS:1000527" name="highest observed m/z" value="769.255798339843977"/>
            <userParam name="filter string" type="xsd:string" value="ITMS + c NSI d w Full ms2 457.72@cid35.00 [115.00-930.00]"/>
            <userParam name="preset scan configuration" type="xsd:string" value="2"/>
            <scanList count="1">
                <cvParam cvRef="MS" accession="MS:1000795" name="no combination" />
                <scan >
                    <cvParam cvRef="MS" accession="MS:1000016" name="scan start time" value="1503.96166992188" unitAccession="UO:0000010" unitName="second" unitCvRef="UO" />
                    <userParam name="[Thermo Trailer Extra]Monoisotopic M/Z:" type="xsd:double" value="457.723968505858977"/>
                    <scanWindowList count="1">
                        <scanWindow>
                            <cvParam cvRef="MS" accession="MS:1000501" name="scan window lower limit" value="115" unitAccession="MS:1000040" unitName="m/z" unitCvRef="MS" />
                            <cvParam cvRef="MS" accession="MS:1000500" name="scan window upper limit" value="930" unitAccession="MS:1000040" unitName="m/z" unitCvRef="MS" />
                        </scanWindow>
                    </scanWindowList>
                </scan>
            </scanList>
            <precursorList count="1">
                <precursor>
                    <isolationWindow>
                        <cvParam cvRef="MS" accession="MS:1000827" name="isolation window target m/z" value="457.723968505859" unitAccession="MS:1000040" unitName="m/z" unitCvRef="MS" />
                        <cvParam cvRef="MS" accession="MS:1000828" name="isolation window lower offset" value="1.5" unitAccession="MS:1000040" unitName="m/z" unitCvRef="MS" />
                        <cvParam cvRef="MS" accession="MS:1000829" name="isolation window upper offset" value="0.75" unitAccession="MS:1000040" unitName="m/z" unitCvRef="MS" />
                    </isolationWindow>
                    <selectedIonList count="1">
                        <selectedIon>
                            <cvParam cvRef="MS" accession="MS:1000744" name="selected ion m/z" value="457.723968505859" unitAccession="MS:1000040" unitName="m/z" unitCvRef="MS" />
                            <cvParam cvRef="MS" accession="MS:1000041" name="charge state" value="2" />
                            <cvParam cvRef="MS" accession="MS:1002815" name="inverse reduced ion mobility" value="1.078628" unitAccession="MS:1002814" unitName="volt-second per square centimeter"/>
                        </selectedIon>
                    </selectedIonList>
                    <activation>
                        <cvParam cvRef="MS" accession="MS:1000133" name="collision-induced dissociation" />
                        <cvParam cvRef="MS" accession="MS:1000045" name="collision energy" value="35.0"/>
                    </activation>
                </precursor>
            </precursorList>
            <binaryDataArrayList count="2">
                <binaryDataArray encodedLength="1088">
                    <cvParam cvRef="MS" accession="MS:1000514" name="m/z array" unitAccession="MS:1000040" unitName="m/z" unitCvRef="MS" />
                    <cvParam cvRef="MS" accession="MS:1000523" name="64-bit float" />
                    <cvParam cvRef="MS" accession="MS:1000576" name="no compression" />
                    <binary>AAAAoExpYkAAAACA3MpkQAAAAACph2VAAAAAAE4wZkAAAACAlMdmQAAAAECZAmdAAAAAwP9jaEAAAADgj4ZoQAAAAGC7HWlAAAAAAOXFaUAAAADg+4dqQAAAAMC1pmpAAAAA4IGFa0AAAACAaUZsQAAAACBzYW1AAAAAANCjbUAAAACAQ6duQAAAAIDsxG5AAAAAQKIlb0AAAACA5z9vQAAAAIDuw29AAAAAAJQicEAAAAAg9UZwQAAAAKCeVHBAAAAAIInEcEAAAACAcs5wQAAAAOA6BHFAAAAAADoOcUAAAAAgfcRxQAAAAOA68nFAAAAAoPExckAAAADATKVyQAAAAMC10nJAAAAAwBJHc0AAAAAA7FNzQAAAAIAYkXNAAAAAgJzRc0AAAABgE2R0QAAAAMCrc3RAAAAAgE+zdEAAAAAAhMR0QAAAAIC64XRAAAAA4Cf/dEAAAADgy3B1QAAAAMCVgnVAAAAAoDugdUAAAACAX/Z1QAAAAAAAB3ZAAAAAgO4XdkAAAABAqEJ2QAAAAIDp8nZAAAAAIAgRd0AAAACggzR3QAAAAODwT3dAAAAAIHJsd0AAAAAA4YJ3QAAAAGC91ndAAAAAAL3id0AAAADg0xZ4QAAAAOA5NXhAAAAAYDaPeEAAAACgK7p4QAAAACCm0XhAAAAA4GHkeEAAAADgyPJ4QAAAAOB5/3hAAAAAoFtNeUAAAADA8H15QAAAAGAHtXlAAAAAoD7HeUAAAAAAEtR5QAAAAGCx5XlAAAAA4NEJekAAAAAgtVN6QAAAACDCX3pAAAAAIAqmekAAAACg4OR6QAAAAGDymnxAAAAAICV/fUAAAAAgd6Z9QAAAAKDYA4BAAAAAoCoVgEAAAACA/kOAQAAAAKCpYoBAAAAA4MycgEAAAADA3DyBQAAAAKCbrIFAAAAAoPC6gUAAAADgV22CQAAAACABY4NAAAAAQE+qg0AAAADA0vKDQAAAAEDz+oNAAAAAoIxrhEAAAADg6euEQAAAAIAuDIVAAAAAoOwjhUAAAACgZUuFQAAAAADdm4VAAAAAoCzrh0AAAABgYvWHQAAAAOALCohA</binary>
                </binaryDataArray>
                <binaryDataArray encodedLength="544">
                    <cvParam cvRef="MS" accession="MS:1000515" name="intensity array" unitAccession="MS:1000131" unitName="number of detector counts" unitCvRef="MS"/>
                    <cvParam cvRef="MS" accession="MS:1000521" name="32-bit float" />
                    <cvParam cvRef="MS" accession="MS:1000576" name="no compression" />
                    <binary>3FlbQDg/ZUB8w3FAV2fMQMiOnkCXfP4/T2I2QC6qskAnhOZA/NU2QCc2QEAI1UhAQcAbQRrziUBmHq5AXutSQWZDbkAZGWdAzt6lQYNptUDSFDNBoY4IQAYaQEDeT7Q/16HGP9GtXUCITrQ/Rxu0Pzhc6j9mpjZAX1X8P7tPQ0AqxS5BZTzZPye+m0B7Sa5AfPsPQRr/W0CYwBRBwDh3QMAmtD/nq6E/bJHGPxJ9UUDsy/dAoCYMQRM2a0BkAR9Boo5pQMV0VEArYu5A4kaMQAyTI0BQPRJAML3TQCKVCED85+tArObGP1BVP0EtJuVAdyKAQFjctkFQa2NBixMTQXyyjUFX8eo/IHelQTdFcEFo1zZAhagsQAO53EBIugRB0M+gQfhBgkH0MsJAbGlIQZXg+EHe6CZBsbA2QHMHOECtW6BAjE2oQUpZckBasZ1AtKl3QEZYIUHkip1AQX7TQPqF60GNuaE/USk2QGLF40Im65ZAmXqlQBGuSUC70KBAAneMQeK3aEB87MVA5NigQE/Wb0BO475A</binary>
                </binaryDataArray>
            </binaryDataArrayList>
        </spectrum>
        "#;
        let mut spectra = MzMLReader::with_file_id(0).parse(s.as_bytes()).await?;

        assert_eq!(spectra.len(), 1);
        let s = spectra.pop().unwrap();

        assert_eq!(s.id, "spectrum=2442");
        assert_eq!(s.ms_level, 2);
        assert_eq!(s.representation, Representation::Centroid);
        assert_eq!(s.precursors.len(), 1);
        assert_eq!(s.precursors[0].charge, Some(2));
        assert!((s.precursors[0].mz - 457.723968) < 0.0001);
        assert!(match s.precursors[0].inverse_ion_mobility {
            Some(x) => (x - 1.0786) < 0.0001,
            None => false,
        });
        assert_eq!(
            s.precursors[0].isolation_window,
            Some(Tolerance::Da(-1.5, 0.75))
        );
        assert!((s.scan_start_time - 25.066).abs() < 0.0001);
        assert_eq!(s.ion_injection_time, 0.0);
        assert_eq!(s.intensity.len(), s.mz.len());
        Ok(())
    }

    #[tokio::test]
    async fn parse_spectrum_issue_117() -> Result<(), MzMLError> {
        // The issue was that some converters write the ion mobility as part of the selected ion (as in the last test)
        // and some write it as part of the scan, as in this test. This test checks that it can be read
        // fom the scan section.
        let s = r#"
        <spectrum id="spectrum=8678309" index="8678309" defaultArrayLength="102" dataProcessingRef="dp_sp_1">
            <cvParam cvRef="MS" accession="MS:1000127" name="centroid spectrum" />
            <cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="2" />
            <cvParam cvRef="MS" accession="MS:1000294" name="mass spectrum" />
            <cvParam cvRef="MS" accession="MS:1000130" name="positive scan" />
            <cvParam cvRef="MS" accession="MS:1000504" name="base peak m/z" value="638.352905273437955"/>
            <cvParam cvRef="MS" accession="MS:1000505" name="base peak intensity" value="113.885513305664006"/>
            <cvParam cvRef="MS" accession="MS:1000285" name="total ion current" value="793.395202636718977"/>
            <userParam name="filter string" type="xsd:string" value="ITMS + c NSI d w Full ms2 457.72@cid35.00 [115.00-930.00]"/>
            <scanList count="1">
                <cvParam cvRef="MS" accession="MS:1000795" name="no combination" />
                <scan >
                    <cvParam cvRef="MS" accession="MS:1000016" name="scan start time" value="1503.96166992188" unitAccession="UO:0000010" unitName="second" unitCvRef="UO" />
                    <cvParam cvRef="MS" accession="MS:1002815" name="inverse reduced ion mobility" value="1.078628" unitAccession="MS:1002814" unitName="volt-second per square centimeter"/>
                    <userParam name="[Thermo Trailer Extra]Monoisotopic M/Z:" type="xsd:double" value="457.723968505858977"/>
                    <scanWindowList count="1">
                        <scanWindow>
                            <cvParam cvRef="MS" accession="MS:1000501" name="scan window lower limit" value="115" unitAccession="MS:1000040" unitName="m/z" unitCvRef="MS" />
                            <cvParam cvRef="MS" accession="MS:1000500" name="scan window upper limit" value="930" unitAccession="MS:1000040" unitName="m/z" unitCvRef="MS" />
                        </scanWindow>
                    </scanWindowList>
                </scan>
            </scanList>
            <precursorList count="1">
                <precursor>
                    <isolationWindow>
                        <cvParam cvRef="MS" accession="MS:1000827" name="isolation window target m/z" value="457.723968505859" unitAccession="MS:1000040" unitName="m/z" unitCvRef="MS" />
                        <cvParam cvRef="MS" accession="MS:1000828" name="isolation window lower offset" value="1.5" unitAccession="MS:1000040" unitName="m/z" unitCvRef="MS" />
                        <cvParam cvRef="MS" accession="MS:1000829" name="isolation window upper offset" value="0.75" unitAccession="MS:1000040" unitName="m/z" unitCvRef="MS" />
                    </isolationWindow>
                    <selectedIonList count="1">
                        <selectedIon>
                            <cvParam cvRef="MS" accession="MS:1000744" name="selected ion m/z" value="457.723968505859" unitAccession="MS:1000040" unitName="m/z" unitCvRef="MS" />
                            <cvParam cvRef="MS" accession="MS:1000041" name="charge state" value="2" />
                        </selectedIon>
                    </selectedIonList>
                    <activation>
                        <cvParam cvRef="MS" accession="MS:1000133" name="collision-induced dissociation" />
                        <cvParam cvRef="MS" accession="MS:1000045" name="collision energy" value="35.0"/>
                    </activation>
                </precursor>
            </precursorList>
            <binaryDataArrayList count="2">
                <binaryDataArray encodedLength="1088">
                    <cvParam cvRef="MS" accession="MS:1000514" name="m/z array" unitAccession="MS:1000040" unitName="m/z" unitCvRef="MS" />
                    <cvParam cvRef="MS" accession="MS:1000523" name="64-bit float" />
                    <cvParam cvRef="MS" accession="MS:1000576" name="no compression" />
                    <binary>AAAAoExpYkAAAACA3MpkQAAAAACph2VAAAAAAE4wZkAAAACAlMdmQAAAAECZAmdAAAAAwP9jaEAAAADgj4ZoQAAAAGC7HWlAAAAAAOXFaUAAAADg+4dqQAAAAMC1pmpAAAAA4IGFa0AAAACAaUZsQAAAACBzYW1AAAAAANCjbUAAAACAQ6duQAAAAIDsxG5AAAAAQKIlb0AAAACA5z9vQAAAAIDuw29AAAAAAJQicEAAAAAg9UZwQAAAAKCeVHBAAAAAIInEcEAAAACAcs5wQAAAAOA6BHFAAAAAADoOcUAAAAAgfcRxQAAAAOA68nFAAAAAoPExckAAAADATKVyQAAAAMC10nJAAAAAwBJHc0AAAAAA7FNzQAAAAIAYkXNAAAAAgJzRc0AAAABgE2R0QAAAAMCrc3RAAAAAgE+zdEAAAAAAhMR0QAAAAIC64XRAAAAA4Cf/dEAAAADgy3B1QAAAAMCVgnVAAAAAoDugdUAAAACAX/Z1QAAAAAAAB3ZAAAAAgO4XdkAAAABAqEJ2QAAAAIDp8nZAAAAAIAgRd0AAAACggzR3QAAAAODwT3dAAAAAIHJsd0AAAAAA4YJ3QAAAAGC91ndAAAAAAL3id0AAAADg0xZ4QAAAAOA5NXhAAAAAYDaPeEAAAACgK7p4QAAAACCm0XhAAAAA4GHkeEAAAADgyPJ4QAAAAOB5/3hAAAAAoFtNeUAAAADA8H15QAAAAGAHtXlAAAAAoD7HeUAAAAAAEtR5QAAAAGCx5XlAAAAA4NEJekAAAAAgtVN6QAAAACDCX3pAAAAAIAqmekAAAACg4OR6QAAAAGDymnxAAAAAICV/fUAAAAAgd6Z9QAAAAKDYA4BAAAAAoCoVgEAAAACA/kOAQAAAAKCpYoBAAAAA4MycgEAAAADA3DyBQAAAAKCbrIFAAAAAoPC6gUAAAADgV22CQAAAACABY4NAAAAAQE+qg0AAAADA0vKDQAAAAEDz+oNAAAAAoIxrhEAAAADg6euEQAAAAIAuDIVAAAAAoOwjhUAAAACgZUuFQAAAAADdm4VAAAAAoCzrh0AAAABgYvWHQAAAAOALCohA</binary>
                </binaryDataArray>
                <binaryDataArray encodedLength="544">
                    <cvParam cvRef="MS" accession="MS:1000515" name="intensity array" unitAccession="MS:1000131" unitName="number of detector counts" unitCvRef="MS"/>
                    <cvParam cvRef="MS" accession="MS:1000521" name="32-bit float" />
                    <cvParam cvRef="MS" accession="MS:1000576" name="no compression" />
                    <binary>3FlbQDg/ZUB8w3FAV2fMQMiOnkCXfP4/T2I2QC6qskAnhOZA/NU2QCc2QEAI1UhAQcAbQRrziUBmHq5AXutSQWZDbkAZGWdAzt6lQYNptUDSFDNBoY4IQAYaQEDeT7Q/16HGP9GtXUCITrQ/Rxu0Pzhc6j9mpjZAX1X8P7tPQ0AqxS5BZTzZPye+m0B7Sa5AfPsPQRr/W0CYwBRBwDh3QMAmtD/nq6E/bJHGPxJ9UUDsy/dAoCYMQRM2a0BkAR9Boo5pQMV0VEArYu5A4kaMQAyTI0BQPRJAML3TQCKVCED85+tArObGP1BVP0EtJuVAdyKAQFjctkFQa2NBixMTQXyyjUFX8eo/IHelQTdFcEFo1zZAhagsQAO53EBIugRB0M+gQfhBgkH0MsJAbGlIQZXg+EHe6CZBsbA2QHMHOECtW6BAjE2oQUpZckBasZ1AtKl3QEZYIUHkip1AQX7TQPqF60GNuaE/USk2QGLF40Im65ZAmXqlQBGuSUC70KBAAneMQeK3aEB87MVA5NigQE/Wb0BO475A</binary>
                </binaryDataArray>
            </binaryDataArrayList>
        </spectrum>
        "#;
        let mut spectra = MzMLReader::with_file_id(0).parse(s.as_bytes()).await?;

        assert_eq!(spectra.len(), 1);
        let s = spectra.pop().unwrap();
        assert!(match s.precursors[0].inverse_ion_mobility {
            Some(x) => (x - 1.0786) < 0.0001,
            None => false,
        });

        // The rest of these assertions just make sure the integrity of the spectrum is maintained
        assert_eq!(s.id, "spectrum=8678309");
        assert_eq!(s.ms_level, 2);
        assert_eq!(s.representation, Representation::Centroid);
        assert_eq!(s.precursors.len(), 1);
        assert_eq!(s.precursors[0].charge, Some(2));
        assert!((s.precursors[0].mz - 457.723968) < 0.0001);
        assert_eq!(
            s.precursors[0].isolation_window,
            Some(Tolerance::Da(-1.5, 0.75))
        );
        assert!((s.scan_start_time - 25.066).abs() < 0.0001);
        assert_eq!(s.ion_injection_time, 0.0);
        assert_eq!(s.intensity.len(), s.mz.len());
        Ok(())
    }

    #[tokio::test]
    async fn parse_spectrum_issue_210() -> Result<(), MzMLError> {
        // Handle cases where both isolation window target m/z is set and different than selected ion m/z
        let s = r#"
        <spectrum id="spectrum=8678309" index="8678309" defaultArrayLength="102" dataProcessingRef="dp_sp_1">
            <cvParam cvRef="MS" accession="MS:1000127" name="centroid spectrum" />
            <cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="2" />
            <precursorList count="1">
                <precursor>
                    <isolationWindow>
                        <cvParam cvRef="MS" accession="MS:1000827" name="isolation window target m/z" value="457.75" unitAccession="MS:1000040" unitName="m/z" unitCvRef="MS" />
                        <cvParam cvRef="MS" accession="MS:1000828" name="isolation window lower offset" value="1.5" unitAccession="MS:1000040" unitName="m/z" unitCvRef="MS" />
                        <cvParam cvRef="MS" accession="MS:1000829" name="isolation window upper offset" value="0.75" unitAccession="MS:1000040" unitName="m/z" unitCvRef="MS" />
                    </isolationWindow>
                    <selectedIonList count="1">
                        <selectedIon>
                            <cvParam cvRef="MS" accession="MS:1000744" name="selected ion m/z" value="457.723968505859" unitAccession="MS:1000040" unitName="m/z" unitCvRef="MS" />
                            <cvParam cvRef="MS" accession="MS:1000041" name="charge state" value="2" />
                        </selectedIon>
                    </selectedIonList>
                </precursor>
            </precursorList>
        </spectrum>
        "#;
        let mut spectra = MzMLReader::with_file_id(0).parse(s.as_bytes()).await?;

        assert_eq!(spectra.len(), 1);
        let s = spectra.pop().unwrap();
        assert!((s.precursors[0].mz - 457.723968) < 0.0001);
        assert_eq!(
            s.precursors[0].isolation_window,
            Some(Tolerance::Da(-1.5, 0.75))
        );

        // Check different ordering of fields in mzML
        let s = r#"
        <spectrum id="spectrum=8678309" index="8678309" defaultArrayLength="102" dataProcessingRef="dp_sp_1">
            <cvParam cvRef="MS" accession="MS:1000127" name="centroid spectrum" />
            <cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="2" />
            <precursorList count="1">
                <precursor>
                    <selectedIonList count="1">
                        <selectedIon>
                            <cvParam cvRef="MS" accession="MS:1000744" name="selected ion m/z" value="457.723968505859" unitAccession="MS:1000040" unitName="m/z" unitCvRef="MS" />
                            <cvParam cvRef="MS" accession="MS:1000041" name="charge state" value="2" />
                        </selectedIon>
                    </selectedIonList>
                    <isolationWindow>
                        <cvParam cvRef="MS" accession="MS:1000827" name="isolation window target m/z" value="457.75" unitAccession="MS:1000040" unitName="m/z" unitCvRef="MS" />
                        <cvParam cvRef="MS" accession="MS:1000828" name="isolation window lower offset" value="1.5" unitAccession="MS:1000040" unitName="m/z" unitCvRef="MS" />
                        <cvParam cvRef="MS" accession="MS:1000829" name="isolation window upper offset" value="0.75" unitAccession="MS:1000040" unitName="m/z" unitCvRef="MS" />
                    </isolationWindow>
                </precursor>
            </precursorList>
        </spectrum>
        "#;
        let mut spectra = MzMLReader::with_file_id(0).parse(s.as_bytes()).await?;

        assert_eq!(spectra.len(), 1);
        let s = spectra.pop().unwrap();
        assert!((s.precursors[0].mz - 457.723968) < 0.0001);
        assert_eq!(
            s.precursors[0].isolation_window,
            Some(Tolerance::Da(-1.5, 0.75))
        );

        // Check fallback keeping iso window m/z of fields in mzML
        let s = r#"
        <spectrum id="spectrum=8678309" index="8678309" defaultArrayLength="102" dataProcessingRef="dp_sp_1">
            <cvParam cvRef="MS" accession="MS:1000127" name="centroid spectrum" />
            <cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="2" />
            <precursorList count="1">
                <precursor>
                    <isolationWindow>
                        <cvParam cvRef="MS" accession="MS:1000827" name="isolation window target m/z" value="457.75" unitAccession="MS:1000040" unitName="m/z" unitCvRef="MS" />
                        <cvParam cvRef="MS" accession="MS:1000828" name="isolation window lower offset" value="1.5" unitAccession="MS:1000040" unitName="m/z" unitCvRef="MS" />
                        <cvParam cvRef="MS" accession="MS:1000829" name="isolation window upper offset" value="0.75" unitAccession="MS:1000040" unitName="m/z" unitCvRef="MS" />
                    </isolationWindow>
                    <selectedIonList count="1">
                        <selectedIon>
                            <cvParam cvRef="MS" accession="MS:1000041" name="charge state" value="2" />
                        </selectedIon>
                    </selectedIonList>
                </precursor>
            </precursorList>
        </spectrum>
        "#;
        let mut spectra = MzMLReader::with_file_id(0).parse(s.as_bytes()).await?;

        assert_eq!(spectra.len(), 1);
        let s = spectra.pop().unwrap();
        assert!((s.precursors[0].mz - 457.75) < 0.0001);
        assert_eq!(
            s.precursors[0].isolation_window,
            Some(Tolerance::Da(-1.5, 0.75))
        );
        Ok(())
    }

    // Exercises referenceableParamGroup handling (issue #232):
    // - `ms level`/centroid pulled from a spectrum-scope ref group
    // - a <scanWindow> ref group (scan-range limits) that sage ignores
    // - an inline isolation (quad) window that the scan-window ref must not disturb
    #[tokio::test]
    async fn parse_referenceable_param_group_issue_232() -> Result<(), MzMLError> {
        let s = r#"
        <mzML>
        <referenceableParamGroupList count="2">
            <referenceableParamGroup id="SpectrumParams">
            <cvParam cvRef="MS" accession="MS:1000580" name="MSn spectrum" />
            <cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="2" />
            <cvParam cvRef="MS" accession="MS:1000127" name="centroid spectrum" />
            </referenceableParamGroup>
            <referenceableParamGroup id="ScanWindowParams">
            <cvParam cvRef="MS" accession="MS:1000501" name="scan window lower limit" unitAccession="MS:1000040" unitCvRef="MS" unitName="m/z" value="140" />
            <cvParam cvRef="MS" accession="MS:1000500" name="scan window upper limit" unitAccession="MS:1000040" unitCvRef="MS" unitName="m/z" value="1750" />
            </referenceableParamGroup>
        </referenceableParamGroupList>
        <run>
            <spectrumList count="1">
            <spectrum id="dia=1" index="0" defaultArrayLength="3">
                <referenceableParamGroupRef ref="SpectrumParams" />
                <cvParam cvRef="MS" accession="MS:1000285" name="total ion current" value="100.0" />
                <scanList count="1">
                <scan>
                    <cvParam cvRef="MS" accession="MS:1000016" name="scan start time" unitAccession="UO:0000031" unitCvRef="MS" unitName="minute" value="1.5" />
                    <scanWindowList count="1">
                    <scanWindow>
                        <referenceableParamGroupRef ref="ScanWindowParams" />
                    </scanWindow>
                    </scanWindowList>
                </scan>
                </scanList>
                <precursorList count="1">
                <precursor>
                    <isolationWindow>
                        <cvParam cvRef="MS" accession="MS:1000827" name="isolation window target m/z" unitAccession="MS:1000040" unitCvRef="MS" unitName="m/z" value="404.5" />
                        <cvParam cvRef="MS" accession="MS:1000828" name="isolation window lower offset" unitAccession="MS:1000040" unitCvRef="MS" unitName="m/z" value="5" />
                        <cvParam cvRef="MS" accession="MS:1000829" name="isolation window upper offset" unitAccession="MS:1000040" unitCvRef="MS" unitName="m/z" value="5" />
                    </isolationWindow>
                    <selectedIonList count="1">
                        <selectedIon>
                        <cvParam cvRef="MS" accession="MS:1000744" name="selected ion m/z" unitAccession="MS:1000040" unitCvRef="MS" unitName="m/z" value="404.5" />
                        </selectedIon>
                    </selectedIonList>
                </precursor>
                </precursorList>
                <binaryDataArrayList count="2">
                <binaryDataArray encodedLength="16">
                    <cvParam cvRef="MS" accession="MS:1000514" name="m/z array" unitCvRef="MS" unitAccession="MS:1000040" unitName="m/z" />
                    <cvParam cvRef="MS" accession="MS:1000523" name="64-bit float" />
                    <cvParam cvRef="MS" accession="MS:1000576" name="no compression" />
                    <binary>AAAAAAAALkAAAAAAAAA0QAAAAAAAADlA</binary>
                </binaryDataArray>
                <binaryDataArray encodedLength="16">
                    <cvParam cvRef="MS" accession="MS:1000515" name="intensity array" unitCvRef="MS" unitAccession="MS:1000131" unitName="number of detector counts" />
                    <cvParam cvRef="MS" accession="MS:1000523" name="64-bit float" />
                    <cvParam cvRef="MS" accession="MS:1000576" name="no compression" />
                    <binary>AAAAAAAAWUAAAAAAAABZQAAAAAAAAFlA</binary>
                </binaryDataArray>
                </binaryDataArrayList>
            </spectrum>
            </spectrumList>
        </run>
        </mzML>
        "#;

        let mut spectra = MzMLReader::with_file_id(0).parse(s.as_bytes()).await?;
        assert_eq!(spectra.len(), 1);
        let s = spectra.pop().unwrap();

        assert_eq!(s.id, "dia=1");
        assert_eq!(s.ms_level, 2, "ms level from spectrum ref group");
        assert_eq!(s.representation, Representation::Centroid);
        assert_eq!(s.mz.len(), 3);
        assert_eq!(s.intensity.len(), 3);
        assert_eq!(s.precursors.len(), 1);
        assert!((s.precursors[0].mz - 404.5).abs() < 1e-4);
        assert_eq!(
            s.precursors[0].isolation_window,
            Some(Tolerance::Da(-5.0, 5.0)),
            "inline isolation window must survive the scanWindow ref group"
        );
        Ok(())
    }
}
