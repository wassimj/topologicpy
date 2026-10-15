# TopologicPy source layout

Keep new implementation helpers as private methods in their respective public API class files. Avoid adding standalone underscore-prefixed Python helper modules under `src/topologicpy` unless explicitly requested. The directory should consist mainly of the main API class modules. Existing unrelated modules do not need to be reorganized as part of a new change.
