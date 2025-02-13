import Pkg
Pkg.activate(".")

# all adds necessary
Pkg.add(name="TensorOperations", version="4.1.1")
Pkg.add(name="TensorKit", version="0.12.7")
Pkg.add(name="MPSKit", version="0.11.3")
Pkg.add(name="MPSKitModels", version="0.3.5")
Pkg.add(name="JLD2", version="0.5.11")
Pkg.add(name="Polynomials", version="4.0.11")
Pkg.add(name="Optim", version="1.10.0")

println("packages added")