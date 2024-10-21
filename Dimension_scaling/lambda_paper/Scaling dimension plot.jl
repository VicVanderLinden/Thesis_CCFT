##Scaling dimension plot

using JLD2
using Plots
using Polynomials

datasector0 = [5.4796-8.1251im, 21.984-9.6931im, 23.4532-11.1809im]
datasector1 = [2.16395-1.63599im, 17.2026-5.14224im, 18.076-11.41066im]


v = datasector1[end]/(2*pi*(1.1336−0.0205im))
v = 2.8810 −0.7091im
println(v)
testsector0 = datasector0/(2*pi*v)
testsector1 = datasector1/(2*pi*v)

actual_sector0 = [0.4656−0.2245im, 1.4656−0.2245im, 1.908-0.599im]
actual_sector1 = [0.1336−0.0205im, 1.111 −0.170im, 1.1336−0.0205im]

q = plot(; xlabel="charge", ylabel="Real Δn",title="Dimension scaling")
plot!(q,[0,0,0],real(testsector0),seriestype=:scatter,label="results",markeralpha=[0.5,0.5,0.5])
plot!(q,[0,0,0],real(actual_sector0),seriestype=:scatter,label="theoretical",markeralpha=[0.5,0.5,0.5])
plot!(q,[1,1,1],real(testsector1),seriestype=:scatter,label="results",markeralpha=[0.5,0.5,0.5])
plot!(q,[1,1,1],real(actual_sector1),seriestype=:scatter,label="theoretical",markeralpha=[0.5,0.5,0.5])
savefig(q,"Real QuasiparticleAnsatz 3Dimensionscaling D50_v.png")

p=plot(; xlabel="charge", ylabel="Im Δn",title="Dimension scaling")
plot!(p,[0,0,0],real(-im.*testsector0),seriestype=:scatter,label="results",markeralpha=[0.5,0.5,0.5])
plot!(p,[0,0,0],real(-im*actual_sector0),seriestype=:scatter,label="theoretical",markeralpha=[0.5,0.5,0.5])
plot!(p,[1,1,1],real(-im*testsector1),seriestype=:scatter,label="results",markeralpha=[0.5,0.5,0.5])
plot!(p,[1,1,1],real(-im*actual_sector1),seriestype=:scatter,label="theoretical",markeralpha=[0.5,0.5,0.5])
savefig(p,"Im QuasiparticleAnsatz 3Dimensionscaling D50_v.png")