@info "about to install packages"
import Pkg
Pkg.activate(".")
Pkg.instantiate()

@info "packages installed"