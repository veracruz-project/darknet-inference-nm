# Darknet inference native module for Veracruz

This is a Darknet inference [native module for Veracruz](https://github.com/veracruz-project/veracruz/discussions/577).
It is meant to be executed by the [native module sandboxer](https://github.com/veracruz-project/native-module-sandboxer) in a sandbox environment everytime a WebAssembly program invokes it.  
Just like any native module, it is an entry point to a more complex library and only exposes preselected high-level features to the programs invoking it.

This native module takes an execution configuration file encoding a `DarknetInferenceService` structure.
It then performs inference on the model with the given input image and outputs a list of detections ordered by descending probability.
Training is not supported.
