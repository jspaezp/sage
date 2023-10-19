use crate::{read_and_execute, Error};
use sage_core::spectrum::RawSpectrum;
use tokio::io::AsyncReadExt;

pub fn read_mzml<S: AsRef<str>>(
    s: S,
    file_id: usize,
    signal_to_noise: Option<u8>,
) -> Result<Vec<RawSpectrum>, Error> {
    read_and_execute(s, |bf| async move {
        Ok(crate::mzml::MzMLReader::with_file_id(file_id)
            .set_signal_to_noise(signal_to_noise)
            .parse(bf)
            .await?)
    })
}

pub fn read_tdf<S: AsRef<str>>(s: S, file_id: usize) -> Result<Vec<RawSpectrum>, Error> {
    let res = crate::tdf::TdfReader::default().parse(s, file_id);
    match res {
        Ok(t) => Ok(t),
        Err(e) => Err(Error::TDF(e)),
    }
}

pub fn read_fasta<S>(
    path: S,
    decoy_tag: S,
    generate_decoys: bool,
) -> Result<sage_core::fasta::Fasta, Error>
where
    S: AsRef<str>,
{
    read_and_execute(path, |mut bf| async move {
        let mut contents = String::new();
        bf.read_to_string(&mut contents)
            .await
            .map_err(crate::Error::IO)?;
        Ok(sage_core::fasta::Fasta::parse(
            contents,
            decoy_tag.as_ref(),
            generate_decoys,
        ))
    })
}

pub fn read_json<S, T>(path: S) -> Result<T, Error>
where
    S: AsRef<str>,
    T: for<'de> serde::Deserialize<'de>,
{
    read_and_execute(path, |mut bf| async move {
        let mut contents = String::new();
        bf.read_to_string(&mut contents).await?;
        Ok(serde_json::from_str(&contents)?)
    })
}
