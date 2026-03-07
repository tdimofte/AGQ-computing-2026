cd(@__DIR__)
using Pkg
Pkg.activate(".")

using Plots, Alert

x = range(0, 2π, 100)
y = sin.(x)

plot(x, y, label = "sin(x)")

alert("Done")