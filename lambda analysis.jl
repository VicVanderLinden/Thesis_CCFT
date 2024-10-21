### lambda analysis
using PlotlyJS

p = PlotlyJS.plot(PlotlyJS.contour(   z=broadcast(log,broadcast(abs,results)),
    x=real(-im*test_values),
        y=real(test_values),fill=true,colorbar=attr(
            title="log(|gε'|)", # title here
            titleside="top",
            titlefont=attr(
                size=14,
                family="Arial, sans-serif"
            )
        )),Layout(title=attr(text = "L=$L",x = 0.5))
)
println(results)
savefig(p,"lambda_estimation_D50,L$L,N4dif.png")  