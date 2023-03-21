//! A native module for ML inference on Darknet.
//! Takes an input image and various parameters, feeds it to the model and
//! outputs a list of detected objects.
//!
//! ## Authors
//!
//! The Veracruz Development Team.
//!
//! ## Licensing and copyright notice
//!
//! See the `LICENSE_MIT.markdown` file in the Veracruz root directory for
//! information on licensing and copyright.

use darknet::{BBox, Detection, Image, Network};
use serde::Deserialize;
use std::cmp::Ordering;
use std::fmt::Write as _;
use std::fs::{read_to_string, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// Module's API.
#[derive(Deserialize, Debug)]
pub(crate) struct DarknetInferenceService {
    /// Path to the input (image) to be fed to the network.
    input_path: PathBuf,
    /// Path to the model's configuration.
    cfg_path: PathBuf,
    /// Path to the Darknet-compatible model (weights file).
    model_path: PathBuf,
    /// Path to the labels file containing all the objects that can be detected.
    labels_path: PathBuf,
    /// Path to the output file containing the result of the prediction.
    output_path: PathBuf,
    /// Threshold above which an object is considered detected.
    objectness_threshold: f32,
    /// Threshold above which a class is considered detected assuming objectness
    /// within the detection box. Darknet internally sets class probabilities to
    /// 0 if they are below the objectness threshold, so this should be above it
    /// to make any difference.
    class_threshold: f32,
    /// Hierarchical threshold. Only used in YOLO9000, a model able to detect
    /// hierarchised objects.
    hierarchical_threshold: f32,
    /// Intersection-over-union threshold. Used to eliminate irrelevant
    /// detection boxes.
    iou_threshold: f32,
    /// Whether the image should be letterboxed, i.e. padded while preserving
    /// its aspect ratio, or resized, before being fed to the model.
    letterbox: bool,
}

impl DarknetInferenceService {
    /// Create a new service, with empty internal state.
    pub fn new() -> Self {
        Self {
            input_path: PathBuf::new(),
            cfg_path: PathBuf::new(),
            model_path: PathBuf::new(),
            labels_path: PathBuf::new(),
            output_path: PathBuf::new(),
            objectness_threshold: 0.0,
            class_threshold: 0.0,
            hierarchical_threshold: 0.0,
            iou_threshold: 0.0,
            letterbox: true,
        }
    }

    /// Try to parse input into a DarknetInferenceService structure.
    /// An attacker may inject malformed paths but that should be caught by the
    /// VFS when attempting to access the corresponding files.
    /// The input image is resized or letterboxed (depending on the `letterbox`
    /// parameter) before being fed to the model, which guarantees dimensions
    /// match.
    fn try_parse(&mut self, input: &[u8]) -> anyhow::Result<bool> {
        let deserialized_input: DarknetInferenceService = match postcard::from_bytes(&input) {
            Ok(o) => o,
            Err(_) => return Ok(false),
        };
        *self = deserialized_input;
        Ok(true)
    }

    /// The core service. It loads the model pointed by `model_path` with the
    /// configuration in `cfg_path` and the labels defined in `labels_path`,
    /// then feeds the input read from `input_path` to the model, and writes the
    /// result to the file at `output_path`.
    fn infer(&mut self) -> anyhow::Result<()> {
        let DarknetInferenceService {
            input_path,
            cfg_path,
            model_path,
            labels_path,
            output_path,
            objectness_threshold,
            class_threshold,
            hierarchical_threshold,
            iou_threshold,
            letterbox,
        } = self;

        // Load network and labels
		println!("loading network...");
        let mut net = Network::load(cfg_path, Some(model_path), false)?;
        let object_labels = read_to_string(labels_path)?
            .lines()
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>();

        // Run inference
        let image = Image::open(input_path)?;
		println!("running inference on image...");
        let detections = net.predict(
            &image,
            *objectness_threshold,
            *hierarchical_threshold,
            *iou_threshold,
            *letterbox,
        );

        // Apply class threshold and map detected objects to labels
        let mut labeled_detections: Vec<(usize, (Detection, f32, &String))> = detections
            .iter()
            .flat_map(|det| {
                det.best_class(Some(*class_threshold))
                    .map(|(class_index, prob)| (det, prob, &object_labels[class_index]))
            })
            .enumerate()
            .collect();

        // Sort labeled detections by descending probability
        labeled_detections.sort_by(|a, b| {
            let (_, (_, prob_a, _)) = a;
            let (_, (_, prob_b, _)) = b;
            if prob_b > prob_a {
                Ordering::Greater
            } else if prob_b < prob_a {
                Ordering::Less
            } else {
                Ordering::Equal
            }
        });

        // Write result to output path
        let mut output = String::new();
        for (_, (detection, prob, label)) in labeled_detections {
            let BBox { x, y, w, h } = detection.bbox();
            write!(
                output,
                "{}\t{:.2}%\tx: {}\ty: {}\tw: {}\th: {}\n",
                label,
                prob * 100.0,
                x,
                y,
                w,
                h,
            )?
        }
        println!("writing results...");
        let mut file = File::create(Path::new("/").join(output_path))?;
        file.write_all(&output.into_bytes())?;

        Ok(())
    }
}

fn main() -> anyhow::Result<()> {
    let mut service = DarknetInferenceService::new();

    // Read input from execution configuration file
    println!("opening execution configuration file...");
    let mut f = File::open("/execution_config")?;
    let mut input = Vec::new();
    println!("reading execution configuration file...");
    f.read_to_end(&mut input)?;
    println!("parsing input...");
    service.try_parse(&input)?;
    service.infer()
}
