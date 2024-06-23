// The Entrypoint of the public Crate.
// Eventaully, the user is gonna create a WaveGraph, that stores the Data either in the
// GraphModel- or WaveModel-State, on being a modified Petgraph, the other a QWT-based
// Graph stored in a WaveletMatrix and a BitMap.
// Depending on the Operations executed, the State of the Model has to be changed.

mod graphmodel;
pub use graphmodel::GraphModel;
mod wavemodel;
use serde::{Deserialize, Serialize};
pub use wavemodel::WaveModel;

#[derive(Serialize, Deserialize)]
pub enum QWT {
    QWT256(qwt::QWT256<usize>),
    QWT512(qwt::QWT512<usize>),
}
