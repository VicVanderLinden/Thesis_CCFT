

Q = 2
using JLD2
using LaTeXStrings
using KrylovKit
include("Potts-Operators & Hamiltonian.jl")
include("Env_adapt.jl")
Vp = Vect[ZNIrrep{Q}](sector=>1 for sector in 0:Q-1)
_,_,W = weyl_heisenberg_matrices(Q)
P   = TensorMap(W,ℂ^Q←ℂ^Q)


function dot_nonHermitian(bra::FiniteMPS{T}, ket::FiniteMPS{T}) where {T}
        N = MPSKit.check_length(bra, ket)
        N_half = N ÷ 2
        return tr(entanglement_Hamiltonian(N_half, bra, ket))
    end
function infinite_potts(lambda;symytry = true,J=1,h=1,Q=5)
    if symytry
    dat0 = reshape((P*sum((-h * potts_spin_shift(; q = Q,k=j)) for j in 1:1:Q-1)*P').data, (Q,Q))
    dat1 = reshape(((P ⊗ P)*sum((-J * potts_phase(; q=Q,k=j)) for j in 1:1:Q-1)*(P' ⊗ P')).data, (Q,Q,Q,Q))
    dat2 = reshape(((P ⊗ P) * sum(sum(potts_phase_shift_combined(;q=Q,k=l,p=j) for l in 1:1:Q-1) for j in 1:1:Q-1) *(P'⊗P')).data, (Q,Q,Q,Q))
    H0 = @mpoham (sum(TensorMap(dat0,Vp←Vp){i} for i in -Inf:Inf)) ### Potts
    H1 = @mpoham (sum(TensorMap(dat1, Vp⊗Vp←Vp⊗Vp){i,i+1}  for i in -Inf:Inf))
    H2 =  @mpoham lambda* sum(TensorMap(dat2,Vp⊗Vp←Vp⊗Vp){i,i+1} for i in -Inf:Inf)
    return H0 + H1 + H2
    else 
        dat0 = reshape((sum((-h * potts_spin_shift(; q = Q,k=j)') for j in 1:1:Q-1)).data, (Q,Q))
        dat1 = reshape((sum((-J * potts_phase(; q=Q,k=j)') for j in 1:1:Q-1)).data, (Q,Q,Q,Q))
        dat2 = reshape((sum(sum(potts_phase_shift_combined(;q=Q,k=l,p=j)' for l in 1:1:Q-1) for j in 1:1:Q-1)).data, (Q,Q,Q,Q))
        H0 = @mpoham (sum(TensorMap(dat0,ℂ^Q←ℂ^Q){i} for i in -Inf:Inf)) ### Potts
        H1 = @mpoham (sum(TensorMap(dat1,ℂ^Q⊗ℂ^Q←ℂ^Q⊗ℂ^Q){i,i+1}  for i in -Inf:Inf))
        H2 =  @mpoham lambda* (sum(TensorMap(dat1,ℂ^Q⊗ℂ^Q←ℂ^Q⊗ℂ^Q){i,i+1} for i in -Inf:Inf))
        return H0 + H1 + H2
    end
    end
function run_sim(Q,D)
        d = D[1]
        H = infinite_potts(0,Q = 2)
        ψ_right = InfiniteMPS(Vp,Vect[ZNIrrep{Q}](sector=>d for sector in 0:Q-1)) 
        # (ψ_right, envs, delta) = find_groundstate(ψ_right, H, VUMPS(maxiter = 200,tol=1e-8, alg_eigsolve =MPSKit.Defaults.alg_eigsolve(; ishermitian=false)))
        d_before = d
        for (i,d) in enumerate(D[1:end])
            @info "chi: $d"
            #trscheme = truncbelow(1-8))
            ψ_right,envs = changebonds(ψ_right,H,OptimalExpand(trscheme = truncdim(d-d_before)))
            @show space(ψ_right.C[1])
            (ψ_right, envs , delta) = find_groundstate(ψ_right, H, VUMPS(maxiter = 1000,tol=1e-10, alg_eigsolve =MPSKit.Defaults.alg_eigsolve(; ishermitian=false)), envs)
            save_object("GS ISING VUMPS for inf,D=$d.jld2",ψ_right)
            # ψ_right,_ = time_evolve(ψ_right, H,0:0.1:5,TDVP())
            # println("D = $d, 5s",expectation_value(ψ_right,H))
            # ψ_right,_ = time_evolve(ψ_right, H,0:0.1:15,TDVP())
            # println("D = $d,20s",expectation_value(ψ_right,H))
            d_before = d
        end
    end

    # ψ_right = load_object("GS sym VUMPS for inf,D=$d.jld2")
    # ψ_left = load_object("GS nonsym star for inf D=$d.jld2")
    # ############################################################# MEHTOD 2 working with MPStricks https://arxiv.org/html/2311.18733v2 #######
    # pos = 0
    # bra = ψ_right
    # N = MPSKit.check_length(bra, bra)
    # ρ_left = isomorphism(left_virtualspace(bra, 1),left_virtualspace(bra, 1))
    # ρ_right = isomorphism(right_virtualspace(bra, N),right_virtualspace(bra, N))
    # T_left = MPSKit.TransferMatrix([TensorMap(conj(bra.AL[l].data),space(bra.AL[l])) for l in 1:pos],bra.AL[1:pos]);
    # T_right = MPSKit.TransferMatrix([TensorMap(conj(bra.AR[l].data),space(bra.AR[l])) for l in (pos+1):N],bra.AR[(pos+1):end]);
    # ρ_left = ρ_left*T_left;
    # ρ_right = T_right*ρ_right;
    # conj_C = TensorMap(conj(bra.C[pos].data),space(bra.C[pos]))
    # @plansor ρA[-1; -2] := ρ_left[1; 2] * (conj_C[2; 3]) * ρ_right[3; -2] * conj(bra.C[pos][1; -1])
    # println("rho achieved")
    # ρA /= tr( ρA)
    # val ,vec= eig(ρA)
    # values = [ lambda for lambda in val.data]
    # # values = values./sum(values)
    # ent = sum(   real(lambda)!=0 ? -lambda*log(Complex(lambda)) : 0 for lambda in values)
    # push!(full_e,ent)
    # levels = 10
    # # eigvec.data = eigvec.data/eigvec[1];
    # val.data ./= val.data[1];
    # ES = zeros(ComplexF64,5,levels)
    # for i=0:4
  
    #     ES[i+1,:] = sort(-log.(block(val,ZNIrrep{5}(i)).diag);by=x -> real(x))[1:levels]
    # end
    # ES ./= ES[1,2]

    # p = plot(title = latexstring("|ψ_{right}><ψ_{right}*| D = $d"),size=(600,500),xlabel="sector",ylabel="Entanglement spectrum",legend=false)

    # for i=0:4
    #     scatter!(p,i*ones(levels),real(ES[i+1,:]),marker=:cross)
    # end
    # for j=1:5
    #     for i=1:levels
    #         annotate!(p,[(j-1.08+0.08*i,real(ES[j,i]),text(latexstring("$i"),8))])
    #     end
    # end
    
    # display(p)


function entanglement_Hamiltonian_alt(H,bra::InfiniteMPS{T},ket::InfiniteMPS{T}) where {T}
    # println(ket.AL[1].data)
    GLs,GRs = MPSKit.initialize_environments(ket,H,bra)
    envs = InfiniteEnvironments(GLs, GRs)
    # GLs = compute_leftenvs!_alt(envs,bra,H,ket,GMRES()) ### might be not needed but its starting xo ###linspace not defined is what its giving me i use arnoldi?
    # println(GLs)
    Te = TransferMatrix(bra.AL, ket.AL)
    λ1, envs.GLs[1] = fixedpoint(flip(Te), envs.GLs[1], :LM, Arnoldi()) #### why only one? because largest????
    # GRs = compute_rightenvs!_alt(envs,bra,H,ket,GMRES())  ########## are these even nessecary?
    Te = TransferMatrix(bra.AR, ket.AR) 
    λ2, envs.GRs[1] = fixedpoint(Te, envs.GRs[1], :LM, Arnoldi())
    # println(GRs[1])
    ############## this ######################""
    @tensor ρA[a,c] := bra.C[1][a,b] * conj(ket.C[1][c,b])
    # println(ρA)
    ρA *=  abs(λ1) * abs(λ2)
    ρA = TensorMap(ρA.data,space(bra.C[1]) )
    ############ or this? "########################################
    # println(space( GLs[2]))
    # @plansor ρA[a,b] := GLs[2][c; d] * (ket.C[1][d; e]) * GRs[2][e; a] * conj(bra.C[1][c; b]) 
    
    return ρA
end


function make_S_IFV_XI(D)
    
    plt = plot(title = latexstring("Entropy scaling D = $D"),size=(600,500),xlabel="ξ",ylabel="Re(S)",label=" ",legend=true,xscale=:log)
    ent = []
    xi = []
    for d in D
       
        ψ_right = load_object("Finite_entanglement_scaling/GS VUMPS for inf,D=$d.jld2") 
        H = infinite_potts(0.079+0.60im)   
        alt = TensorMap(conj(( ψ_right.AL[1].data)),space(ψ_right.AL[1]))
        alt_C = TensorMap(conj(( ψ_right.C[1].data)),space(ψ_right.C[1]))
        alt_AR = TensorMap(conj(( ψ_right.AR[1].data)),space(ψ_right.AR[1]))
        ket = InfiniteMPS([alt,],[alt_AR,],[alt_C,])
        H = entanglement_Hamiltonian_alt(H,ψ_right,ket);
        H =H / tr(H);
        val,vec =  eig(H,(1,),(2,))
        # val.data ./= val.data[1]
        println(val.data)
        push!(ent,sum(   real(lambda)>0 ? -lambda*log((lambda)) : 0 for lambda in val.data) )  
        push!(xi,correlation_length_alt(ψ_right)) ## doesnt work for finite? transfermatrix somehow?
        
    end
    scatter!(plt,xi,real(ent))
    display(plt)
    plat = plot(title = latexstring("Entropy scaling D = $D"),size=(600,500),xlabel="ξ",ylabel="Im(S)",label=" ",legend=true,xscale=:log)
    scatter!(plat,xi,real(-1im.*ent))
    display(plat)
end



function make_S_IFV_XI_ISING()
    
    plt = plot(title = "Entropy Scaling",size=(600,500),xlabel="Ln(ξ)",ylabel="Re(S)",label=" ",legend=true)
    ent_p = []
    xi_p = []
    for d in 3:30
       Q = 5
        ψ_right = load_object("Finite_entanglement_scaling/GS ISING VUMPS for inf,D=$d.jld2") 
        H = infinite_potts(0,Q = 2)   
        H = entanglement_Hamiltonian_alt(H,ψ_right,ψ_right);
        H =H / tr(H);
        val,vec =  eig(H,(1,),(2,))
        # val.data ./= val.data[1]
        println(val.data)
        push!(ent_p,sum(   real(lambda)>0 ? -lambda*log((lambda)) : 0 for lambda in val.data) )  
        push!(xi_p,correlation_length_alt(ψ_right)) ## doesnt work for finite? transfermatrix somehow?
    end
    fit = linear_fit(log.(xi_p), ent_p)
    println(fit)
    scatter!(plt,log.(xi_p),real(ent_p),label = "Ising",m=:diamond)
    x = 1:0.1:6.7
    c = real(round(6*fit[2],digits = 5))
    plot!(plt, x,(real(fit[2].*x .+ fit[1])),label = "c =$c ")

    # ent_p = []
    # xi_p = []
    # for d in 20:30
    #     Q = 5
    #      ψ_right = load_object("Finite_entanglement_scaling/GS nonsym ISING VUMPS for inf,D=$d.jld2") 
    #      H = infinite_potts(0,Q = 2)   
    #      H = entanglement_Hamiltonian_alt(H,ψ_right,ψ_right);
    #      H =H / tr(H);
    #      val,vec =  eig(H,(1,),(2,))
    #      # val.data ./= val.data[1]
    #      println(val.data)
    #      push!(ent_p,sum(   real(lambda)>0 ? -lambda*log((lambda)) : 0 for lambda in val.data) )  
    #      push!(xi_p,correlation_length_alt(ψ_right)) ## doesnt work for finite? transfermatrix somehow?
    #  end
    #  fit = linear_fit(log.(xi_p), ent_p)
    #  println(fit)
    #  scatter!(plt,log.(xi_p),real(ent_p),label = "nonsym Ising",m=:diamond)
    #  c = real(round(6*fit[2],digits = 5))
    #  plot!(plt, x,(real(fit[2].*x .+ fit[1])),label = "NonSym c =$c ")
end



function make_S_IFV_XI_potts(D)
    
    plt = plot(title = "Entropy Scaling",size=(600,500),xlabel="ln(ξ)",ylabel="Re(S)",label=" ",legend=true)
    ent_p = []
    xi_p = []
    ent_c = []
    xi_c = []
    x = -0.5:0.1:5
    for d in 1:38
        if d == 5  
        else
        ψ_right = load_object("Finite_entanglement_scaling/GS VUMPS for inf,D=$d.jld2") 
        H = infinite_potts(0.079+0.60im)   
        alt = TensorMap(conj(( ψ_right.AL[1].data)),space(ψ_right.AL[1]))
        alt_C = TensorMap(conj(( ψ_right.C[1].data)),space(ψ_right.C[1]))
        alt_AR = TensorMap(conj(( ψ_right.AR[1].data)),space(ψ_right.AR[1]))
        ket = InfiniteMPS([alt,],[alt_AR,],[alt_C,])
        H = entanglement_Hamiltonian_alt(H,ψ_right,ket);
        # H =H / tr(H);
        val,vec =  eig(H,(1,),(2,))
        # val.data ./= val.data[1]
        println(val.data)
        push!(ent_c,sum(   real(lambda)>0 ? -lambda*log((lambda)) : 0 for lambda in val.data) ) 
        #push!(ent_c,real(entropy( ψ_right)[1]) )
        alt = TensorMap(conj(( ψ_right.AL[1].data)),space(ψ_right.AL[1]))
        alt_C = TensorMap(conj(( ψ_right.C[1].data)),space(ψ_right.C[1]))
        alt_AR = TensorMap(conj(( ψ_right.AR[1].data)),space(ψ_right.AR[1]))
        ket = InfiniteMPS([alt,],[alt_AR,],[alt_C,])
        # GLs,GRs = MPSKit.initialize_environments(ket,ψ_right)
        Te = TransferMatrix( ψ_right.AR, ket.AR) 
        sector = ZNIrrep{Q}(0)
        init = randomize!(similar(ψ_right.AR[1], left_virtualspace(ket, 1),
        ℂ[typeof(sector)](sector => 1)' * left_virtualspace(ψ_right, 1)))
        λ2,_,_ = eigsolve(Te, init,3,:LM, tol= 1e-10)
        println( λ2)
        push!(xi_c,(1 ./log(abs(λ2[1])/abs(λ2[2]))))  
        end     
    end
    scatter!(plt,log.(xi_c),real(ent_c),label="CCFT",m =:xcross,markersize = 7,color = "green")
    ent_c = []
    xi_c = []
    for d in 1:38
        ψ_right = load_object("Finite_entanglement_scaling/GS VUMPS for inf,D=$d.jld2") 
        H = infinite_potts(0.079+0.60im)   
        alt = TensorMap(conj(( ψ_right.AL[1].data)),space(ψ_right.AL[1]))
        alt_C = TensorMap(conj(( ψ_right.C[1].data)),space(ψ_right.C[1]))
        alt_AR = TensorMap(conj(( ψ_right.AR[1].data)),space(ψ_right.AR[1]))
        ket = InfiniteMPS([alt,],[alt_AR,],[alt_C,])
        H = entanglement_Hamiltonian_alt(H,ψ_right,ket);
        H =H / tr(H);
        val,vec =  eig(H,(1,),(2,))
        # val.data ./= val.data[1]
        println(val.data)
        #push!(ent_c,sum(   real(lambda)>0 ? -lambda*log((lambda)) : 0 for lambda in val.data) ) 
        push!(ent_c,real(entropy( ψ_right)[1]) )
        alt = TensorMap(conj(( ψ_right.AL[1].data)),space(ψ_right.AL[1]))
        alt_C = TensorMap(conj(( ψ_right.C[1].data)),space(ψ_right.C[1]))
        alt_AR = TensorMap(conj(( ψ_right.AR[1].data)),space(ψ_right.AR[1]))
        ket = InfiniteMPS([alt,],[alt_AR,],[alt_C,])
        # GLs,GRs = MPSKit.initialize_environments(ket,ψ_right)
        Te = TransferMatrix( ψ_right.AR, ket.AR) 
        sector = ZNIrrep{Q}(0)
        init = randomize!(similar(ψ_right.AR[1], left_virtualspace(ket, 1),
        ℂ[typeof(sector)](sector => 1)' * left_virtualspace(ψ_right, 1)))
        λ2,_,_ = eigsolve(Te, init,3,:LM, tol= 1e-10)
        println( λ2)
        push!(xi_c,(1 ./log(abs(λ2[1])/abs(λ2[2]))))     
    end
    scatter!(plt,log.(xi_c),real(ent_c),label="CCFT (naive)",m =:xcross,markersize = 7,color = "blue")
    fit_x = [xi_c[i] for i in [1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38]]
    fit_y = [ent_c[i] for i in [1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38]]
    fit = linear_fit(log.(fit_x), fit_y)
    c = real(round(6*fit[2],digits = 5))
    plot!(plt, x,(real(fit[2].*x .+ fit[1])),label = "c =$c ",color = "blue ")

    
    #c = real(round(6*fit[2],digits = 5))
    #plot!(plt, x,(real(fit[2].*x .+ fit[1])),label = "c =$c ",color = "green")
    for d in 1:38
        if d == 3 || d == 2 || d == 4 || d ==7 || d ==8 || d ==25 
        else
        Q = 5
         ψ_right = load_object("Finite_entanglement_scaling/GS POTTS VUMPS for inf,D=$d.jld2") 
         H = infinite_potts(0)   
         H = entanglement_Hamiltonian_alt(H,ψ_right,ψ_right);
         H =H / tr(H);
         val,vec =  eig(H,(1,),(2,))
         # val.data ./= val.data[1]
         println(val.data)
         #push!(ent_p,sum(   real(lambda)>0 ? -lambda*log((lambda)) : 0 for lambda in val.data) )  
         push!(ent_p,real(entropy( ψ_right)[1]) )
         push!(xi_p,correlation_length_alt(ψ_right)) ## doesnt work for finite? transfermatrix somehow?
        end
     end
   
     fit = linear_fit(log.(xi_p), ent_p)
     c = real(round(6*fit[2],digits = 5))
     for d in 1:38
        Q = 5
         ψ_right = load_object("Finite_entanglement_scaling/GS POTTS VUMPS for inf,D=$d.jld2") 
         H = infinite_potts(0)   
         H = entanglement_Hamiltonian_alt(H,ψ_right,ψ_right);
         H =H / tr(H);
         val,vec =  eig(H,(1,),(2,))
         # val.data ./= val.data[1]
         println(val.data)
         #push!(ent_p,sum(   real(lambda)>0 ? -lambda*log((lambda)) : 0 for lambda in val.data) )  
         push!(ent_p,real(entropy( ψ_right)[1]) )
         push!(xi_p,correlation_length_alt(ψ_right)) ## doesnt work for finite? transfermatrix somehow?
     end
     scatter!(plt,log.(xi_p),real(ent_p),label = "Potts Q5",m=:diamond,color = "red")
     plot!(plt, x,(real(fit[2].*x .+ fit[1])),label = "c =$c ",color = "red")
    # plt = plot(title = latexstring("Entropy Scaling"),size=(600,500),xlabel="ξ",ylabel="Im(S)",label=" ",legend=true,xaxis=:log)
    # scatter!(plt,xi,real(-1im.*ent))
    display(plt)
end

function make_S_IFV_D_potts(D)

    plt = plot(title = "Entanglement S",size=(600,500),xlabel="χ",ylabel="Re(S)",legend=true)
    ent_c = []
    ent_c_alt = []
    ent_p = []
    for d in D
        ψ_right = load_object("Finite_entanglement_scaling/GS VUMPS for inf,D=$d.jld2")    
        H = infinite_potts(0.079+0.60im)   
        alt = TensorMap(conj(( ψ_right.AL[1].data)),space(ψ_right.AL[1]))
        alt_C = TensorMap(conj(( ψ_right.C[1].data)),space(ψ_right.C[1]))
        alt_AR = TensorMap(conj(( ψ_right.AR[1].data)),space(ψ_right.AR[1]))
        ket = InfiniteMPS([alt,],[alt_AR,],[alt_C,])
        H = entanglement_Hamiltonian_alt(H,ψ_right,ket);
        H =H / tr(H);
        val,vec = eig(H)
        push!(ent_c,sum(   real(lambda)>0 ? -lambda*log((lambda)) : 0 for lambda in val.data) )
        push!(ent_c_alt,real(entropy( ψ_right)[1]) )

    end
    scatter!(plt,5 .*D,real(ent_c),label="CCFT",m =:xcross,markersize = 7,color="green")
    scatter!(plt,5 .*D,real(ent_c_alt),label="CCFT (naive)",m =:xcross,markersize = 7,color="blue")
    for d in D
        ψ_right = load_object("Finite_entanglement_scaling/GS POTTS VUMPS for inf,D=$d.jld2")   
        push!(ent_p,entropy(ψ_right)[1])
    end
    scatter!(plt,5 .*D,real(ent_p),label="Potts Q5",m=:diamond,color="red")
   
    display(plt)
    
    plat = plot(title = "Entanglement S",size=(600,500),xlabel="χ",ylabel="Im(S)",legend=true)
    scatter!(plat,5 .*D,imag(ent_c),label="CCFT",m =:xcross,markersize = 7)
  
    scatter!(plat,5 .*D,fill(0,length(ent_p)),label="Potts Q5",m = :diamond)
    display(plat)
end
function make_S_IFV_D(D)

    plt = plot(title = "entaglement Infinite CCFT",size=(600,500),xlabel="D",ylabel="Re(S)",legend=true)
    ent = []
    ent_false = []
    for d in D
        ψ_right = load_object("Finite_entanglement_scaling/GS VUMPS for inf,D=$d.jld2")    
        H = infinite_potts(0.079+0.60im)   
        alt = TensorMap(conj(( ψ_right.AL[1].data)),space(ψ_right.AL[1]))
        alt_C = TensorMap(conj(( ψ_right.C[1].data)),space(ψ_right.C[1]))
        alt_AR = TensorMap(conj(( ψ_right.AR[1].data)),space(ψ_right.AR[1]))
        ket = InfiniteMPS([alt,],[alt_AR,],[alt_C,])
        H = entanglement_Hamiltonian_alt(H,ψ_right,ket);
        H =H / tr(H);
        val,vec = eig(H)
        push!(ent,sum(   real(lambda)>0 ? -lambda*log((lambda)) : 0 for lambda in val.data) )
        push!(ent_false,entropy(ψ_right)[1])
    end
    scatter!(plt,D,real(ent),label="own non hermitic function")
    scatter!(plt,D,real(ent_false),label="hermitic function")
    display(plt)
    println(ent)
    println(ent_false)
    plat = plot(title = "entaglement Infinite CCFT",size=(600,500),xlabel="D",ylabel="Im(S)",legend=true)
    scatter!(plat,D,real(-1im.*ent),label="own non hermitic function")
    scatter!(plat,D,fill(0,length(ent)),label="hermitic function")
    display(plat)
end

function make_XI_IFV_D(D)

    plt = plot(title = "ξ Infinite CCFT",size=(600,500),xlabel="D",ylabel="ξ",legend=true)
    xi = []
    xi_false = []
    for d in D
        ψ_right = load_object("Finite_entanglement_scaling/GS VUMPS for inf,D=$d.jld2")    
        
        push!(xi,correlation_length_alt(ψ_right)) ## doesnt work for finite? transfermatrix somehow?
        push!(xi_false,correlation_length(ψ_right))
    end

    scatter!(plt,D,xi,label="own non hermitian function")
    scatter!(plt,D,xi_false,label="hermitian function")
    display(plt)
end
using MPSKit: randomize!
function make_XI_IFV_D_potts(D)

    plt = plot(title = "Correlation length ξ",size=(600,500),xlabel="χ",ylabel="ξ",legend=true)
    xi_p = []
    xi_c = []
    xi_c_alt = []
    for d in D
        ψ_right = load_object("Finite_entanglement_scaling/GS Potts VUMPS for inf,D=$d.jld2")    
        Te = TransferMatrix( ψ_right.AR,ψ_right.AR) 
        sector = ZNIrrep{Q}(0)
        init = randomize!(similar(ψ_right.AR[1], left_virtualspace(ψ_right, 1),
        ℂ[typeof(sector)](sector => 1)' * left_virtualspace(ψ_right, 1)))
        λ2,_,_ = eigsolve(Te, init,3,:LM, tol= 1e-10)

        println( λ2)
        push!(xi_p,correlation_length(ψ_right,tol = 1e-10))
    end
    for d in D

        ψ_right = load_object("Finite_entanglement_scaling/GS VUMPS for inf,D=$d.jld2")    
        alt = TensorMap(conj(( ψ_right.AL[1].data)),space(ψ_right.AL[1]))
        alt_C = TensorMap(conj(( ψ_right.C[1].data)),space(ψ_right.C[1]))
        alt_AR = TensorMap(conj(( ψ_right.AR[1].data)),space(ψ_right.AR[1]))
        ket = InfiniteMPS([alt,],[alt_AR,],[alt_C,])
        # GLs,GRs = MPSKit.initialize_environments(ket,ψ_right)
        Te = TransferMatrix( ψ_right.AR, ket.AR) 
        sector = ZNIrrep{Q}(0)
        init = randomize!(similar(ψ_right.AR[1], left_virtualspace(ket, 1),
        ℂ[typeof(sector)](sector => 1)' * left_virtualspace(ψ_right, 1)))
        λ2,_,_ = eigsolve(Te, init,3,:LM, tol= 1e-10)

        println(λ2)
        push!(xi_c,abs(1 ./log((λ2[1])/(λ2[2]))))

        Te = TransferMatrix( ψ_right.AR, ψ_right.AR) 
        sector = ZNIrrep{Q}(0)
        init = randomize!(similar(ψ_right.AR[1], left_virtualspace(ket, 1),
        ℂ[typeof(sector)](sector => 1)' * left_virtualspace(ψ_right, 1)))
        λ2,_,_ = eigsolve(Te, init,3,:LM, tol= 1e-10)

        println( λ2)
        push!(xi_c_alt,abs(1 ./log((λ2[1])/(λ2[2]))))
    end
    scatter!(plt,5 .*D,xi_c,label= "CCFT",m =:xcross,markersize = 7,color = "green")
    scatter!(plt,5 .*D,xi_c_alt,label="CCFT (naive)",m =:xcross,markersize = 7,color ="blue")
    scatter!(plt,5 .*D,xi_p,label="POTTS Q5",m =:diamond,color ="red")
    # plt = plot(title = "Correlation length ξ",size=(600,500),xlabel="χ",ylabel="ξ",legend=true)
    # xi_p = []
    # xi_c = []
    # for d in  10:30
    #     ψ_right = load_object("Finite_entanglement_scaling/GS nonsym Potts VUMPS for inf,D=$d.jld2")    
    #     Te = TransferMatrix( ψ_right.AR,ψ_right.AR) 
    #     sector = ZNIrrep{Q}(0)
    #     init = randomize!(similar(ψ_right.AR[1], left_virtualspace(ψ_right, 1),
    #     left_virtualspace(ψ_right, 1)))
    #     λ2,_,_ = eigsolve(Te, init,3,:LM, tol= 1e-10)

    #     println( λ2)
    #     push!(xi_p,correlation_length(ψ_right,tol = 1e-10))
    # end
    # for d in 25:55

    #     ψ_right = load_object("Finite_entanglement_scaling/GS nonsym CCFT VUMPS for inf,D=$d.jld2")    
    #     alt = TensorMap(conj(( ψ_right.AL[1].data)),space(ψ_right.AL[1]))
    #     alt_C = TensorMap(conj(( ψ_right.C[1].data)),space(ψ_right.C[1]))
    #     alt_AR = TensorMap(conj(( ψ_right.AR[1].data)),space(ψ_right.AR[1]))
    #     ket = InfiniteMPS([alt,],[alt_AR,],[alt_C,])
    #     # GLs,GRs = MPSKit.initialize_environments(ket,ψ_right)
    #     Te = TransferMatrix( ψ_right.AR, ket.AR) 
    #     sector = ZNIrrep{Q}(0)
    #     init = randomize!(similar(ψ_right.AR[1], left_virtualspace(ket, 1),
    #    left_virtualspace(ψ_right, 1)))
    #     λ2,_,_ = eigsolve(Te, init,3,:LM, tol= 1e-10)

    #     println( λ2)
    #     push!(xi_c,abs(1 ./log((λ2[1])/(λ2[2]))))
    # end
    # scatter!(plt,25:55,xi_c,label="CCFT",m =:circle,color="blue")
    # scatter!(plt, 10:30,xi_p,label="POTTS Q5",m =:utriangle,color = "red")
    display(plt)
    display(plt)
end
function make_f_ifv_D(D)
    H = infinite_potts(0.079+0.06im)
    plt = plot(title = "Free Energy Density CCFT",size=(600,500),xlabel="D",ylabel="Re(f)",legend=true)  
    plat = plot(title = "Free Energy Density CCFT",size=(600,500),xlabel="D",ylabel="Im(f)",legend=true)
 
    f = []
    for d in D
        ψ_right = load_object("Finite_entanglement_scaling/GS VUMPS for inf,D=$d.jld2") 
        af = expectation_value(ψ_right,H)   ### Is this okay? is it not supposed <ψ_left | H | ψ_right>
        push!(f,af)
    end
    plot!(plt,5 .*D,real(f),label = "VUMPS")
    plot!(plat,5 .*D,imag(f),label = "VUMPS")
    display(plt)
    display(plat)
end
function make_f_IFV_L(L)
    H = infinite_potts(0.079+0.06im,syms )
    plt = plot(title = "Free Energy Density CCFT",size=(600,500),xlabel="L",ylabel="Re(f)",legend=true)  
    plat = plot(title = "Free Energy Density CCFT",size=(600,500),xlabel="L",ylabel="Im(f)",legend=true)
 
    xi = []
    # for d in [60,70,80,90]
    
    #     ψ_right = load_object("GS VUMPS for inf,D=$d.jld2") 
    #     f = expectation_value(ψ_right,H)   ### Is this okay? is it not supposed <ψ_left | H | ψ_right>
    #     e = 5*d
    #     plot!(plt,L,fill(real(f),length(L)),label = "VUMPS D = $e",linestyle=:dash)
    #     plot!(plat,L,fill(real(-1im*f),length(L)),label = "VUMPS D = $e",linestyle=:dash)
    # end
    for d in [30,]
    
        ψ_right = load_object("Finite_entanglement_scaling/GS VUMPS for inf,D=$d.jld2") 
        f = expectation_value(ψ_right,H)   ### Is this okay? is it not supposed <ψ_left | H | ψ_right>
        e = 5*d
        plot!(plt,L,fill(real(f),length(L)),label = "VUMPS χ = $e",linestyle=:dash,linewidth = 2)
        plot!(plat,L,fill(real(-1im*f),length(L)),label = "VUMPS χ = $e",linestyle=:dash,linewidth = 2)
    end
    # for d in [60,70,80,90]
    
    #     ψ_right = load_object("GS VUMPS for inf lower tol,D=$d.jld2") 
    #     f = expectation_value(ψ_right,H)   ### Is this okay? is it not supposed <ψ_left | H | ψ_right>
    #     e = 5*d
    #     plot!(plt,L,fill(real(f),length(L)),label = "VUMPS lower tol D = $e",linestyle=:dash)
    #     plot!(plat,L,fill(real(-1im*f),length(L)),label = "VUMPS lower tol D = $e",linestyle=:dash)
    # end
  
  
    # D = 25
    # ψ_right = load_object("Finite_entanglement_scaling/GS sym VUMPS for inf,D=$D.jld2") 
    # f = expectation_value(ψ_right,H)   ### Is this okay? is it not supposed <ψ_left | H | ψ_right>
    # plot!(plt,L,fill(real(f),length(L)),label = "VUMPS D = 125",linestyle=:dash)
    # plot!(plat,L,fill(real(-1im*f),length(L)),label = "VUMPS D = 125",linestyle=:dash)
    energie = []
    for l in L 
        D = 400
        H = Potts_Hamiltonian(l)
        ψ_right = load_object("Finite_entanglement_scaling/GS for L=$l,D=$D.jld2") 
        push!(energie,expectation_value( ψ_right ,H)/l)
    # push!(f,correlation_length(ψ_right)) ## doesnt work for finite? transfermatrix somehow?
    # plot!(plt,D,xi)
    # display(plt)
    end
    scatter!(plt,L,real(energie),label ="DMRG χ = 400",marker = :cross,markersize = 8)
    # energies = []
    # for l in [10,12,14,16,18,20,22] 
    #     D = 600
    #     H = Potts_Hamiltonian(l)
    #     ψ_right = load_object("Finite_entanglement_scaling/GS for L=$l,D=$D.jld2") 
    #     push!(energies,expectation_value( ψ_right ,H)/l)
    # end

    scatter!(plat,L,real(-1im.*energie),label ="DMRG χ = 400",marker = :cross,markersize = 8)
    #scatter!(plt,[10,12,14,16,18,20,22],real(energies),label ="DMRG D = 600",marker = :cross)
    # scatter!(plat,[10,12,14,16,18,20,22],imag(-energies),label ="DMRG D = 600",marker = :cross)
    display(plt)
    display(plat)
end


function make_ES(D;levels=5)
    ES = []
    for d in D
        ψ_right = load_object("Finite_entanglement_scaling/GS VUMPS for inf,D=$d.jld2")
        alt = TensorMap(conj(( ψ_right.AL[1].data)),space(ψ_right.AL[1]))
        alt_C = TensorMap(conj(( ψ_right.C[1].data)),space(ψ_right.C[1]))
        alt_AR = TensorMap(conj(( ψ_right.AR[1].data)),space(ψ_right.AR[1]))
        ket = InfiniteMPS([alt,],[alt_AR,],[alt_C,])
        H = infinite_potts(0.079+0.060im, Q = 5) 
        H = entanglement_Hamiltonian_alt(H,ψ_right,ket);
        H /= tr(H);
        val,vec = eig(H)
        val.data ./= val.data[1];
        ES_D = zeros(ComplexF64,5,levels)
        for i=0:4
    
            ES_D[i+1,:] = sort(-log.(block(val,ZNIrrep{Q}(i)).diag);by=x -> real(x))[1:levels]
            
        end
        
        ES_D ./= ES_D[1,2]
     
        push!(ES,ES_D)
    end
    
        ### plot the scaling in funciton of D
        theoretical  = [[0,2,3,4,4,5,5,6,6,6,6,6],[1,2,3,3,4,4,4,5,5,5+0.3063im - 0.9190im,5+0.3063im - 0.9190im,5].- 0.3063im]
        for j=1:2
            sector = j-1
            p = plot(xlims = [0.17,1 /log.(D[1])+ 0.1],title = "Entanglement spectrum sector $sector",size=(450,600),xlabel="1/ln(χ)",ylabel="Real entanglement spectrum",legend=false)
            for i =1:levels
              
                y = [ES[x][j,i] for x in 1:length(D)]
                fit = linear_fit(1.0./log.(D), 2*y)
                scatter!(p,1 ./log.(5 .*D),2*real(y), markersize = 6,label="$i")
                x = LinRange(0.20,0.46,50)
                #plot!(p,x,real(fit[2].*x .+fit[1]),color="gray")
               
            end
          
            for k in theoretical[j][1:levels]
                plot!(p,[0.18,0.50],[real(k),real(k)],linestyle=:dash,color="black")
                degen = sum([real(m) == real(k) for m in theoretical[j]])
                annotate!(p,[(0.185,real(k) + 0.14,text(latexstring("$degen"),9,color="red"))])
            end
            k = 0
            for i=1:levels
                
                if i>1 && 2*abs(real(ES[1][j,i]) - real(ES[1][j,i-1])) < 0.10
                    if i < 11
                    k += 0.01
                    else
                     k+=0.015
                    end
                else 
                    k = 0
                end
                annotate!(p,[(1 /log.(D[1])+ 0.03 +k,2*real(ES[1][j,i])+ 0.01,text(latexstring("$i"),12))])
          end
            display(p)
        end
        D = D[1:end]
        for j=1:2
            sector = j-1
            p = plot(xlims = [0.17,1 /log.(D[1])+ 0.1],title = "Entanglement spectrum sector $sector",size=(450,600),xlabel="1/ln(χ)",ylabel="Imaginary entanglement spectrum",legend=false)
            for i =1:levels
                
                y = [ES[x][j,i] for x in 1:length(D)]
                plot!(p,1 ./log.(5 .*D),2*real(-1im.*y), markersize = 6,marker=:circle,label="$i")
                x = LinRange(0.05,0.46,50)
                fit = linear_fit(1.0./log.(D), 2*y)
                #plot!(p,x,imag(fit[2]*x .+fit[1]),color="gray")
            end
            k = 0
            for i=1:levels
                if i>1 && 2*abs(real(-1im.*ES[1][j,i]) - real(-1im.*ES[1][j,i-1])) < 0.10
                    if i < 10
                        k += 0.01
                        else
                         k+=0.015
                        end
                else 
                    k = 0
                end
                annotate!(p,[(1 /log.(D[1])+ 0.03+k,2*real(-1im.*ES[1][j,i]) + 0.01,text(latexstring("$i"),12))])
            end
             
            for k in theoretical[j][1:levels]
                plot!(p,[0.18,0.50],[imag(k),imag(k)],linestyle=:dash,color="black")
                degen = sum([imag(m) == imag(k) for m in theoretical[j]])
                annotate!(p,[(0.185,imag(k) + 0.05,text(latexstring("$degen"),9,color="red"))])
            end
            display(p)
        end
    
    
end



function transfer_spectrum_alt(above::InfiniteMPS; below=above, tol=MPSKit.Defaults.tol, num_vals=20,
    sector=first(sectors(oneunit(left_virtualspace(above, 1)))))
    init = MPSKit.randomize!(similar(above.AL[1], left_virtualspace(below, 1),
    ℂ[typeof(sector)](sector => 1)' * left_virtualspace(above, 1)))
    
    
    alt = TensorMap(conj((above.AL[1].data)),space(above.AL[1]))
    alt_C = TensorMap(conj((above.C[1].data)),space(above.C[1]))
    alt_AR = TensorMap(conj((above.AR[1].data)),space(above.AR[1]))
    below = InfiniteMPS([alt,],[alt_AR,],[alt_C,])

    transferspace = fuse(left_virtualspace(above, 1) * left_virtualspace(below, 1)')
    num_vals = min(dim(transferspace, sector), num_vals) # we can ask at most this many values
    eigenvals, eigenvecs, convhist = MPSKit.eigsolve(flip(MPSKit.TransferMatrix(above.AL, below.AL)), init, num_vals, :LM; tol=tol)
    convhist.converged < num_vals &&
        @warn "correlation length failed to converge: normres = $(convhist.normres)"
    return eigenvals
end
function correlation_length_alt(bra::InfiniteMPS{T}) where {T}
   
    spectrum = transfer_spectrum(bra)
    inds = findall(abs.(spectrum) .< 1 - 1e-12)
    length(spectrum) - length(inds) < 2 || @warn "Non-injective mps?"

    spectrum = spectrum[inds]

    e, = marek_gap(spectrum)
    # println(1/e)
    ## or
    spectrum =  sort(-1 ./log.(spectrum);by=x -> real(x),rev=true)
    # println(spectrum[1])
    return abs(spectrum[1])
end

using CurveFit

using Plots
# D = [24,25,26,27,28,29,30,31,32]

# make_S_IFV_D(D)
# make_XI_IFV_D(D)
# make_S_IFV_XI(D)
# syms = true
# # make_f_IFV_L([4,6,8,10,12,14,16,18,20,22,24,26,28])
# # D = [1,2,3,4,6,8,10,11,13,15,17,19]
# # D = 29:50
# # D = [10,15,20]
# D = 1:30 
# syms = true
# run_sim(2,D)

# # D = [12,16,20,24,28]
# D = [20,21,22,23,24,25,26,27,28]

# D = [21,22,23,24,25,27,28,29,31,32,33,34]    
# make_ES(D)
# D = [20,40]
# make_S_IFV_XI_potts(D)
# make_f_ifv_D(D)
# D = 1:38
# make_S_IFV_D_potts(D)
# D = 1:38
# make_XI_IFV_D_potts(D)



# ψ_right = load_object("GS VUMPS for inf,D=5.jld2") 

# println(entanglement_Hamiltonian_alt(ψ_right))


make_S_IFV_XI_ISING()



