##### Complex entanglement entropy for complex conformal field theory #####

## attempt to recreate the entanglement scaling

using Colors

using MPSKit
using MPSKitModels
using TensorKit
using TensorOperations
using Plots
using Polynomials
using LinearAlgebra 
using JLD2
using MPSKit: tsvd
using LsqFit
using LaTeXStrings
Q = 5
Vp = Vect[ZNIrrep{Q}](sector=>1 for sector in 0:Q-1)





function run_sum(L_list,Q,D,syms)
println("start")
for (i,L) in enumerate(L_list)
    H = Potts_Hamiltonian(L,Q=Q,sym=syms,lambda =0.079-0.06im )
    H_dag = Potts_Hamiltonian(L,Q=Q,sym=syms,lambda = 0.079+0.06im)   ## H without lambda is hermitian so should be the same
    if syms 
        ψ₀ = FiniteMPS(L,Vp,Vect[ZNIrrep{Q}](sector=>D for sector in 0:Q-1)) 
    else 
        ψ₀ = FiniteMPS(L,ℂ^Q, ℂ^D)
    end
    (ψ_right, envir , delta) = find_groundstate(ψ₀, H, DMRG(maxiter = 500,tol=1e-14, eigalg =MPSKit.Defaults.alg_eigsolve(; ishermitian=false)))
    (ψ_left, envir , delta) = find_groundstate(ψ₀, H_dag, DMRG(maxiter = 500,tol=1e-14, eigalg =MPSKit.Defaults.alg_eigsolve(; ishermitian=false)))
    if syms
    save_object("GS for L=$L,D=$D.jld2",ψ_right)
    save_object("GS star for L=$L,D=$D.jld2",ψ_left)
    else 
    save_object("GS nonsym for L=$L,D=$D.jld2",ψ_right)
    save_object("GS nonsym star for L=$L,D=$D.jld2",ψ_left)
    end
end
end



function conj_tensor(A,pos)
    
    if pos>0
    dat2 = [TensorMap(conj(B.data),space(B)) for B in [A.AL[1:pos]...,A.AR[pos+1:end]...]]
    else
        dat2 = [TensorMap(conj(B.data),space(B)) for B in [A.AR[pos+1:end]...]]
    end
   
    return FiniteMPS(dat2)
end

function entanglement_Hamiltonian(pos::Int64, bra::FiniteMPS{T}, ket::FiniteMPS{T}) where {T}
    N = MPSKit.check_length(bra, ket)
    ρ_left = isomorphism(storagetype(T),left_virtualspace(bra, 1),left_virtualspace(ket, 1))
    ρ_right = isomorphism(storagetype(T),right_virtualspace(ket, N),right_virtualspace(bra, N))
    T_left = MPSKit.TransferMatrix(ket.AL[1:pos],bra.AL[1:pos]);
    T_right = MPSKit.TransferMatrix(ket.AR[(pos+1):end],bra.AR[(pos+1):end]);
    ρ_left = ρ_left*T_left;
    ρ_right = T_right*ρ_right;

    @plansor ρA[-1; -2] := ρ_left[1; 2] * ket.C[pos][2; 3] * ρ_right[3; -2] * conj(bra.C[pos][1; -1])
    return ρA
end




function entanglement_Hamiltonian_alt(pos::Int64, bra::FiniteMPS{T}) where {T}
    N = MPSKit.check_length(bra, bra)
    ρ_left = isomorphism(storagetype(T),left_virtualspace(bra, 1),left_virtualspace(bra, 1))
    ρ_right = isomorphism(storagetype(T),right_virtualspace(bra, N),right_virtualspace(bra, N))
    T_left = MPSKit.TransferMatrix([TensorMap(conj(bra.AL[l].data),space(bra.AL[l])) for l in 1:pos],bra.AL[1:pos]);
    T_right = MPSKit.TransferMatrix([TensorMap(conj(bra.AR[l].data),space(bra.AR[l])) for l in (pos+1):N],bra.AR[(pos+1):end]);
    ρ_left = ρ_left*T_left;
    ρ_right = T_right*ρ_right;
    conj_C = TensorMap(conj(bra.C[pos].data),space(bra.C[pos]))
    @plansor ρA[-1; -2] := ρ_left[1; 2] * (conj_C[2; 3]) * ρ_right[3; -2] * conj(bra.C[pos][1; -1])
    return ρA
end


using CurveFit

function make_ES(L_list,D;levels=12)
    ES_L = []
    for L in L_list
        ψ_right = load_object("Finite_entanglement_scaling/GS for L=$L,D=$D.jld2")
        H = entanglement_Hamiltonian_alt(L÷2,ψ_right);
        H /= tr(H);
        val,vec = eig(H)
        val.data ./= val.data[1];
        ES = zeros(ComplexF64,5,levels)
        for i=0:4
    
            ES[i+1,:] = sort(-log.(block(val,ZNIrrep{5}(i)).diag);by=x -> real(x))[1:levels]
            
        end
      

        ES ./= (ES[1,2])
        # for i =0:4
        # println(ES[i+1,:])
        # end
        p = plot(title = latexstring("|ψ_{right}><ψ_{left}| L = $L, D = $D"),xlabel="sector",ylabel="Entanglement spectrum",legend=false)

        for i=0:4
            scatter!(p,i*ones(levels),real(ES[i+1,:]),marker=:cross)
        end
        for j=1:5
            for i=1:levels
                annotate!(p,[(j-1.08+0.08*i,real(ES[j,i]),text(latexstring("$i"),12))])
            end
        end
        push!(ES_L,ES)
        # display(p)
    end
    ### plot the scaling in funciton of l 
    theoretical  = [[0,2,3,4,4,5,5,6,6,6-0.9190im,6,6],[1,2,3,3,4,4,4,5,5,5+0.3063im - 0.9190im,5+0.3063im - 0.9190im,5].- 0.3063im]
    for j=1:2
        sector = j-1
        p = plot(xlims = [0.17,1 /log.(L_list[1])+ 0.1],title = "PBC Entanglement spectrum sector $sector",size=(450,600),xlabel="1/ln(L)",ylabel="Real entanglement spectrum",legend=false)
        for i =1:levels
          
            y = [ES_L[x][j,i] for x in 1:length(L_list)]
            fit = linear_fit(1.0./log.(L_list), 2*y)
            scatter!(p,1 ./log.(L_list),2*real(y), markersize = 6,label="$i")
            x = LinRange(0.20,0.46,50)
            plot!(p,x,real(fit[2].*x .+fit[1]),color="gray")
           
        end
      
        for k in theoretical[j][1:levels]
            plot!(p,[0.18,0.50],[real(k),real(k)],linestyle=:dash,color="black")
            degen = sum([real(m) == real(k) for m in theoretical[j]])
            #annotate!(p,[(0.185,real(k) + 0.14,text(latexstring("$degen"),9,color="red"))])
        end
        k = 0
        for i=1:levels
            
            if i>1 && 2*abs(real(ES_L[1][j,i]) - real(ES_L[1][j,i-1])) < 0.10
                if i < 11
                k += 0.01
                else
                 k+=0.015
                end
            else 
                k = 0
            end
            annotate!(p,[(1 /log.(L_list[1])+ 0.03 +k,2*real(ES_L[1][j,i])+ 0.01,text(latexstring("$i"),12))])
      end
        display(p)
    end
    L_list = L_list[1:end]
    for j=1:2
        sector = j-1
        p = plot(xlims = [0.17,1 /log.(L_list[1])+ 0.1],title = "PBC Entanglement spectrum sector $sector",size=(450,600),xlabel="1/ln(L)",ylabel="Imaginary entanglement spectrum",legend=false)
        for i =1:levels
            
            y = [ES_L[x][j,i] for x in 1:length(L_list)]
            plot!(p,1 ./log.(L_list),2*real(-1im.*y), markersize = 6,marker=:circle,label="$i")
            x = LinRange(0.05,0.46,50)
            fit = linear_fit(1.0./log.(L_list), 2*y)
            #plot!(p,x,imag(fit[2]*x .+fit[1]),color="gray")
        end
        k = 0
        for i=1:levels
            if i>1 && 2*abs(real(-1im.*ES_L[1][j,i]) - real(-1im.*ES_L[1][j,i-1])) < 0.10
                if i < 10
                    k += 0.01
                    else
                     k+=0.015
                    end
            else 
                k = 0
            end
            annotate!(p,[(1 /log.(L_list[1])+ 0.03+k,2*real(-1im.*ES_L[1][j,i]) + 0.01,text(latexstring("$i"),12))])
        end
         
        for k in theoretical[j][1:levels]
            plot!(p,[0.18,0.50],[imag(k),imag(k)],linestyle=:dash,color="black")
            degen = sum([imag(m) == imag(k) for m in theoretical[j]])
            #annotate!(p,[(0.185,imag(k) + 0.05,text(latexstring("$degen"),9,color="red"))])
        end
        display(p)
    end

    c1 = colorant"red"
  
    
    c2 = colorant"green"

    
    pm =range(c2, stop=c1, length=levels)
    for j=1:2
        sector = j-1
        p = plot(title = "X-X Entanglement spectrum sector $sector",size=(600,500),xlabel="Im(h)",ylabel="Re(h)",legend=false)
        k = 0
        for i =1:levels
            
            y = [ES_L[x][j,i] for x in 1:length(L_list)]
            plot!(p,real(-1im.*y),real(y),label="$i",marker=:cross,arrow=true)
            if i>1 && 2*abs(ES_L[end][j,i] -ES_L[end][j,i-1]) < 0.2
                k -= 0.04
            else 
                k = 0
            end
            annotate!(p,[(-0.50+k,real(y[end]),text(latexstring("$i"),8))])
        end
      
        display(p)
    end

end





function make_E(L_list,D)
    
    plt = plot(title = latexstring("PBC") *" " *latexstring("χ = $D"),size=(600,500),xlabel="Subsystem L-l",ylabel="Re(S)",legend=true)
    plat = plot(title = latexstring("PBC")  *" " * latexstring("χ = $D"),size=(600,500),xlabel="Subsystem L-l",ylabel="Im(S)",legend=:bottomright)
    for L in L_list
        println(L)
        @. model(l,p) = (p[1]/3)*log((2*L/pi)*sin((pi*l)/L))+p[2]   
        ent = []
        ψ_right = load_object("Finite_entanglement_scaling/GS for L=$L,D=$D.jld2")  
        # ψ_left = load_object("GS star for L=$L,D=$D.jld2")  
        
        for l in 0:L
            H = entanglement_Hamiltonian_alt(l,ψ_right);
        H =H / tr(H);
        eigvec = eig(H)
        values = [ lambda for lambda in eigvec[1].data]
        # values = values./norm
        push!(ent,sum(   real(lambda)!=0 ? -lambda*log(abs((complex(lambda)))) : 0 for lambda in values) )  
        end
       
        scatter!(plt,-Int(L/2):(Int(L/2)),real(ent[1:end]),label = "")
        if L in [20,22,24,26,28,32,48,64]
        scatter!(plat,-Int(L/2):(Int(L/2)),real(1im.*ent[1:end]),label = "")
        end

        p0 = [1.1391,1]
        #f = curve_fit(model, 1:L-1, real(ent[2:end-1]), p0)
        f = curve_fit(model, Int(L/2)-2:Int(L/2)+2, real(ent[Int(L/2)-1:Int(L/2)+3]), p0)
    
        x = 0.95:0.05:L-(0.95)
        values = coef(f)
        c = round( values[1],digits = 5)
        println(  round( values[2],digits = 5))
        plot!(plt,x.-(Int(L/2)),model(x,values),label = "L = $L, Re(c) = $c")
        p0 = [-0.021,0]
        #f = curve_fit(model, 1:L-1, real(-1im.*ent[2:end-1]), p0)
        f = curve_fit(model, Int(L/2)-2:Int(L/2)+2, real(1im.*ent[Int(L/2)-1:Int(L/2)+3]), p0)
    
        values = coef(f)
        c = round( values[1],digits = 5)
        println(  round( values[2],digits = 5))
        println(c)
        if L in [20,22,24,26,28,32,48,64]
         plot!(plat,x.-(Int(L/2)),model(x,values),label = "L = $L, Im(c) = $c i")
        end
        
    end
    display(plt)
    savefig(plt,"ent real PBC 600")
    display(plat)
    savefig(plat,"ent im PBC 600")
end


function make_S_IFV_XI(L_list,D)
    
    plt = plot(title = latexstring("Entanglement scaling L =$L_list, D = $D"),size=(600,500),xlabel="ξ",ylabel="Re(S)",legend=true)
    ent = []
    xi = []
    for L in L_list
       
        ψ_right = load_object("Finite_entanglement_scaling/GS for L=$L,D=$D.jld2")    
        l = Int(L/2)
        H = entanglement_Hamiltonian_alt(l,ψ_right);
        H =H / tr(H);
        eigvec = eig(H,(1,),(2,))
        values = sort([ lambda for lambda in eigvec[1].data],by=x -> real(x),rev=true)
        # values = values./norm
        push!(ent,sum(   real(lambda)>0 ? -lambda*log(abs(lambda)) : 0 for lambda in values) )  
        push!(xi,(1/log(values[1]/values[2])))
        # push!(xi,correlation_length(ψ_right.AL[l] )) 
        
    end
    plot!(real(xi),real(ent))
    println(xi)
    display(plt)
end


function make_conj_plot(L_list,D)
    
    results = []
    results2 = []
    results3 =[]
    plt = plot(title ="Potts Magnitude difference",size=(600,500),xlabel="L",legend=true)
    for L in L_list
        
        l = Int(L/2)
        ψ_right = (load_object("Finite_entanglement_scaling/GS for L=$L,D=$D.jld2") )
        ψ_left = (load_object("Finite_entanglement_scaling/GS star for L=$L,D=$D.jld2"))  
        for l in 1:L

            if l>1
            
            @tensor norm[a,b] := norm[f,d]*conj(ψ_left.AL[l][f,c,a])*(ψ_right.AL[l][d,c,b])
            
            else
                
            @tensor norm[a,b] := conj(ψ_left.AL[1][d,e,a])*(ψ_right.AL[1][d,e,b])
            end
        end
        norm = norm.data[1]
        println(norm)
        push!(results,(abs(norm)))

        for l in 1:L
            alt = TensorMap(conj(ψ_right.AL[l].data),space(ψ_right.AL[l]))
            if l>1
            
            @tensor norm[a,b] := norm[f,d]*conj(alt[f,c,a])*(ψ_right.AL[l][d,c,b])
            
            else
                
            @tensor norm[a,b] := conj(alt[d,e,a])*(ψ_right.AL[1][d,e,b])
            end
        end
        norm = norm.data[1]
        println(norm)
        push!(results2,(abs(norm)))
        for l in 1:L
            alt = TensorMap(conj(ψ_right.AL[l].data),space(ψ_right.AL[l]))
            if l>1
            
            @tensor norm[a,b] := norm[f,d]*conj(alt[f,c,a])*(ψ_left.AL[l][d,c,b])
            
            else
                
            @tensor norm[a,b] := conj(alt[d,e,a])*(ψ_left.AL[1][d,e,b])
            end
        end
        norm = norm.data[1]
        println(norm)
        push!(results3,(abs(norm)))
    end
    plot!(L_list,results, label = "| < ψ_{L} | {ψ}_{R}> |")
    plot!(L_list,results2, label = "| < ψ_{R}* | {ψ}_{R}> |")
    plot!(L_list,results3, label = "| < ψ_{R}* | {ψ}_{L}> |")
    display(plt)
end
using LinearAlgebra
function make_S_and_E(L,D)
     for i in -1:1:1
        ψ_right = load_object("Finite_entanglement_scaling/GS for L=$L,D=$D.jld2") 
        p = plot(size = (800,700),title = "Difference in scaling for L/2 +$i",xlabel = "i",legendfontsize=13,ylims = (10^(-18),1)) 
        l = Int(L/2) +i 
        println(l)
        H = entanglement_Hamiltonian_alt(l,ψ_right);
        H =H / tr(H);
        eigvec = eig(H)
        println(keys(eigvec))
        values = sort([ lambda for lambda in eigvec[1].data],by=x -> (real(x)),rev=true)
        values_alt = []
        for k=1:5
            for val in entanglement_spectrum( ψ_right, l).values[k]
           push!(values_alt,val^2)
            end
        end
        values_balt = []
        for k=1:5
              for val in LinearAlgebra.svdvals(H).values[k]
           push!(values_balt,val^2)
              end
            end
        values_alt = sort(values_alt,by=x -> (real(x)),rev=true)
        values_balt = sort(values_balt,by=x -> (real(x)),rev=true)
        # left = transpose((ψ_right.C[l])')
        #test = [s^2 for s in LinearAlgebra.svdvals(left).values[1]]
        println(size(LinearAlgebra.svdvals(H).values[1]),size(LinearAlgebra.svdvals(H).values[2]))
        #tast = [s^2 for s in LinearAlgebra.svdvals(H)]
        #plot!(1:400,abs.(imag(values[1:400])),label = latexstring("|Im(E_i)|"),yscale=:log10)
        #plot!(p,1:400,abs.(real(values[1:400])),label = latexstring(" | Re(E_i) |"),yscale=:log10)
        plot!(p,1:400,real(values_alt[1:400]).^2,label  =latexstring("s_i^4")*" of "*latexstring("|ψ_R>"),yscale=:log10)
        #plot!(p,1:79,test[1:79],label  =latexstring("s_i^2 |ψ_R*>"))
        plot!(p,1:400,values_balt[1:400],label  =latexstring("s_i^2")*" of "*latexstring("ρ_{RL}"),yscale=:log10)
         display(p)
    end

# savefig(p,"eeee")
end



L = [12,16,18,22,24,26,28]
#  L = 24
D = 400
# Q = 5
# make_S_and_E(L,D)
# make_conj_plot(L,D)
# make_S_IFV_XI(L,D)
# make_E(L,D)
make_ES(L,D)
# make_S_IFV_XI(L,D)



















### my old code 
# function make_E(L_list,D)
    
#     p = plot(title = latexstring("PBC D = $D"),size=(600,500),xlabel="Subsystem L-l",ylabel="Re(S)",legend=true)
#     for L in L_list
#         @. model(l,p) = (p[1]/3)*log((L/pi)*sin((pi*l)/L))+p[2]
#         ent = []
#         ψ_right = load_object("GS for L=$L,D=$D.jld2")  
#         ψ_left = load_object("GS star for L=$L,D=$D.jld2")  
#         for l in 0:L
#         # # for i in 1:L+1
#         # C_R = ψ_right.C[i-1]
#         # C_L = ψ_left.C[i-1]

    
    








#         # if i > 1
#         #     for l in 1:i-1
#         #         if l>1
#         #             @tensor rho[a,b] := rho[f,d]*ψ_left.AL[l][f,c,a]*conj(ψ_right.AL[l][d,c,b])
#         #         else 
#         #             @tensor rho[a,b] := ψ_left.AL[1][d,e,a]*conj(ψ_right.AL[1][d,e,b])
                   
#         #         end
                    
#         #     end 
#         #     M_A = rho
#         # else 
#         #     M_A = 1
#         # end
        
        
#         # if i < L
#         #     for l in L:-1:i
#         #         if l<L
#         #             @tensor rho[a,b] := rho[f,d]*ψ_left.AR[l][a,g,f]*conj(ψ_right.AR[l][b,g,d])
#         #         else  
#         #             @tensor rho[a,b] := ψ_left.AR[L][a,e,c]*conj(ψ_right.AR[L][b,e,c])
#         #         end
              
#         #     end
#         #     M_B = rho
#         # else 
#         #     M_B = 1
#         # end
        
#         # M_A = transpose(M_A)
#         # if M_A ==1
#         #     @tensor rho[a,e] :=  M_B[a,b]*C_R[b,c]*C_L[c,e]
#         # elseif M_B == 1 
#         #     @tensor rho[a,e] :=  C_R[a,c]*M_A[c,d]*C_L[d,e]
#         # else
#         #     @tensor  rho[a,e] :=  M_B[a,b]*C_R[b,c]*M_A[c,d]*C_L[d,e]
#         # end
        
#         # println("rho achieved,$i")
        
#         # ψ_left = conj_tensor(ψ_right,l)
#         H = entanglement_Hamiltonian(l,ψ_right,ψ_left);
#         H =H / tr(H);
#         eigvec = eig(H,(1,),(2,))
#         values = [ lambda for lambda in eigvec[1].data]
#         # values = values./norm
#         push!(ent,sum(   real(lambda)>0 ? -lambda*log(abs(lambda)) : 0 for lambda in values) )  
#         end
#         p0 = [1.1391,0]
#         p = scatter!(-Int(L/2):(Int(L/2)),real(ent[1:end]),label = "")
#         # println(ent[Int(L/2)-:Int(L/2)+5])
#         f = curve_fit(model, 1:L-1, real(ent[2:end-1]), p0)
#         x = 1:0.05:L-1
#         values = coef(f)
#         c = round( values[1],digits = 5)
#         plot!(x.-(Int(L/2)),model(x,values),label = "L =  = $L fit, c = $c")
        
#     end
#     display(p)
# end


#norm = 0
#     println("norm")
#     for l in 1:L
#         if l>1
#             @tensor norm[a,b] := norm[f,d]*conj(ψ_left.AL[l][f,c,a])*ψ_right.AL[l][d,c,b]
          
#         else 
            
#             @tensor norm[a,b] := conj(ψ_left.AL[1][d,e,a])*(ψ_right.AL[1][d,e,b])
           
#         end       
#     end
#     norm = norm.data[1]'
#     println(norm)







#     for i in 1:L+1
#     C_R = ψ_right.C[i-1]
#     C_L = ψ_left.C[i-1]
#     if i > 1
#         for l in 1:i-1
#             if l>1
#                 @tensor rho[a,b] := rho[f,d]*ψ_left.AL[l][f,c,a]*conj(ψ_right.AL[l][d,c,b])
#             else 
#                 @tensor rho[a,b] := ψ_left.AL[1][d,e,a]*conj(ψ_right.AL[1][d,e,b])
               
#             end
                
#         end 
#         M_A = rho
#     else 
#         M_A = 1
#     end
    
    
#     if i < L
#         for l in L:-1:i
#             if l<L
#                 @tensor rho[a,b] := rho[f,d]*ψ_left.AR[l][a,g,f]*conj(ψ_right.AR[l][b,g,d])
#             else  
#                 @tensor rho[a,b] := ψ_left.AR[L][a,e,c]*conj(ψ_right.AR[L][b,e,c])
#             end
          
#         end
#         M_B = rho
#     else 
#         M_B = 1
#     end
    
#     M_A = transpose(M_A)
#     if M_A ==1
#         @tensor rho[a,e] :=  M_B[a,b]*C_R[b,c]*C_L[c,e]
#     elseif M_B == 1 
#         @tensor rho[a,e] :=  C_R[a,c]*M_A[c,d]*C_L[d,e]
#     else
#         @tensor  rho[a,e] :=  M_B[a,b]*C_R[b,c]*M_A[c,d]*C_L[d,e]
#     end
    
#     println("rho achieved,$i")
#     eigvec = eig(rho,(1,),(2,))
#     values = [ lambda for lambda in eigvec[1].data]
#     values = values./norm
#     ent[i] += sum(   real(lambda)!=0 ? -lambda*log(abs(lambda)) : 0 for lambda in values)
#     s = ent[i]
#     println("ent is $s \n") 
#     if i-1 ==Int(L/2)
#         println(i)
#         for bb in blocks(eigvec[1])
#             values = diag(convert(Array,bb[2]))
#             values = values./norm
#             push!(Full_Ent_spectrum, [abs(lambda)!=0 ? -(1/2*pi)*log(abs(lambda)) : 0 for lambda in values])
        
#         end
#     end
# end
# push!(Full_Ent,ent)