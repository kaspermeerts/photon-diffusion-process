using Printf
using Random
using SpecialFunctions
using StaticArrays

mutable struct Epdf
	xs::Vector{Float64}
	bins::Vector{Int}
	min::Float64
	max::Float64
	d::Float64
	num_ensemble::Int
	num_ensemble_planck::Int

	function Epdf(min, max, d, num_ensemble::Int, planck_proportion, ic)
		epdf = new(
			zeros(Float64, num_ensemble),
			zeros(Int, round(Int, (max - min)/d) + 1),
			min,
			max,
			d,
			num_ensemble,
			round(Int, num_ensemble / planck_proportion)
		)
		init!(epdf, ic)
		return epdf
	end
end

function update!(epdf::Epdf)
	fill!(epdf.bins, 0)
	for x in epdf.xs
		i = floor(Int,(x - epdf.min) / epdf.d) + 1
		epdf.bins[clamp(i,begin,end)] += 1
	end
end

function estimate_pdf(epdf::Epdf, x)
	i = floor(Int, (x - epdf.min) / epdf.d) + 1
	return epdf.bins[clamp(i, begin, end)] / (epdf.d*epdf.num_ensemble)
end

function estimate_number(epdf::Epdf, x)
	estimate_pdf(epdf, x) * 2zeta(3) * epdf.num_ensemble / epdf.num_ensemble_planck / x^2
end

function dump_hist(t, epdf::Epdf)
	open(@sprintf("hist%06d.dat", t*100), "w") do io
		for i in eachindex(epdf.bins)
			println(io, epdf.min + ((i-1) + 1/2) * epdf.d, '\t', epdf.bins[i] / (epdf.num_ensemble * epdf.d))
		end
	end
end

function dump_occupation(t, epdf::Epdf)
	open(@sprintf("occu%06d.dat", t*100), "w") do io
		for i in eachindex(epdf.bins)
			println(io,
				epdf.min + ((i-1) + 1/2) * epdf.d, '\t',
				epdf.bins[i] / (epdf.num_ensemble_planck * epdf.d) * 2zeta(3) / (((i-1)+1/2) *epdf.d)^2
			)
		end
	end
end

function init!(epdf::Epdf, ic)
	for i in eachindex(epdf.xs)
		if ic == "uniform"
			epdf.xs[i] = 1 + 8rand()
		elseif ic == "boltzmann"
			epdf.xs[i] = randexp() + randexp() + randexp()
		elseif ic == "planck" || ic == "hotplanck"
			temperature = ic == "hotplanck" ? 2 : 1
			F(x) = 1/(2zeta(3)*temperature) * (x/temperature)^2 / (exp(x/temperature) - 1)
			sample = 0.0
			while !(rand() < F(sample) / 0.26)
				sample = 20rand()
			end
			epdf.xs[i] = sample
		elseif ic == "squared"
			epdf.xs[i] = 6*cbrt(rand())
		elseif ic == "invexp"
			epdf.xs[i] = rand()
		else
			throw("No such ic")
		end
	end
	update!(epdf)
end

struct Sim
	epdf::Epdf
	dt::Float64
	
	b::Float64
	k::Float64
	
	c::Float64
	
	brems::Bool
	
	reset::Bool
	rate::Float64	
end

function tick!(sim::Sim, i)
	epdf = sim.epdf
	dx = epdf.d
	x = epdf.xs[i]
	x0 = 2 * dx
	
	drift = 4x - x^2*(1 + b*x^k)*(1 + estimate_number(epdf,x)) + (x < x0 ? 3c/(x0) : c/x^2)
	
	diffusion = sqrt(2*(x^2 + (x < x0 ? c*x/x0^2 : c/x)))
	
	epdf.xs[i] = x + drift*dt + diffusion*randn()*sqrt(dt)

	if epdf.xs[i] < epdf.min
		#@printf "Underflow %.2f\n" epdf.xs[i]
		epdf.xs[i] = rand()*2*dx
	elseif epdf.xs[i] > epdf.max
		#@printf "Overflow  %.2f\n" epdf.xs[i]
		epdf.xs[i] = x - rand()*2*dx
	end
	
	return
end

function do_brems!(sim::Sim, bin)
	epdf = sim.epdf
	dx = epdf.d
	planck(x) = 1/2zeta(3) * (x^2 / (exp(x) - 1))
	
	bin_goal = epdf.num_ensemble_planck*dx*planck((bin - 1)*dx + dx/2)

	goal_frac, goal_int::Int = modf(bin_goal)
	excess = epdf.bins[bin] - (goal_int + (rand() < goal_frac ? 1 : 0))
	if excess < 0
		#@printf "Adding   %d\n" -excess
		for i in 1:(-excess)
			push!(epdf.xs, (bin - 1 + rand())*dx)
		end
	elseif excess > 0
		#@printf "Removing %d\n" excess
		for i in 1:excess
			for j in eachindex(epdf.xs)
				if (bin - 1) *dx < epdf.xs[j] < bin * dx
					epdf.xs[end], epdf.xs[j] = epdf.xs[j], epdf.xs[end]
					pop!(epdf.xs)
					break
				end
			end
		end
	end
	epdf.num_ensemble -= excess
	epdf.bins[bin] -= excess
	
	return
end

function trajectory(sim::Sim, tf)

	t = 0.0
	t_step = 1.0
	t_goal = 0t_step
	
	times = [t]
	particle_ns = [num_particles]

	start_time = time()
	steps = 0
	while t < tf
		Threads.@threads for j in 1:sim.epdf.num_ensemble
			tick!(sim, j)
		end
		
		update!(sim.epdf)

		if sim.brems
			for brems_bin in 1:2
				do_brems!(sim, brems_bin)
			end
		end
		
		if false && sim.reset
			resets = 0
			for i in eachindex(sim.epdf.xs)
				if rand() < sim.dt*sim.rate
					sim.epdf.xs[i] /= 2
					resets += 1
				end
			end
			update!(sim.epdf)
		end
				
		t += sim.dt
		if t >= t_goal
			t_goal += t_step
			elapsed = time() - start_time
			eta = (tf - t) / (t - 0) * elapsed
			total = (tf - 0) / (t - 0) * elapsed
			epdf = sim.epdf
			@printf("%2.2f / %2.2f ETA %5d %g out of %g: %.2f%% of Planckian value\n", t, tf, eta, epdf.num_ensemble, epdf.num_ensemble_planck, 100*epdf.num_ensemble/epdf.num_ensemble_planck)
			dump_hist(t, epdf)
			dump_occupation(t, epdf)
			open("n.dat", "w") do io
				for i in eachindex(times)
					println(io, times[i], '\t', particle_ns[i] / epdf.num_ensemble_planck)
				end
			end
		end
		push!(times, t)
		push!(particle_ns, sim.epdf.num_ensemble)
	end
	println()

	dump_hist(t, sim.epdf)
	dump_occupation(t, sim.epdf)
	
	open("n.dat", "w") do io
		for i in eachindex(times)
			println(io, times[i], '\t', particle_ns[i] / sim.epdf.num_ensemble_planck)
		end
	end
	
	return
end

cd("data")
foreach(rm,
	filter(
		endswith(".dat"),
		readdir()
	)
)

const Z = 1.10
const num_particles = round(Int, 1_00_000*Z)
const dx = 0.05
const dt = 0.0001
const ic = "hotplanck"
const b = 0.000
const k = 0.000
const c = 0.000
const brems = false
const tf = 100.0
const reset = false
const rate = 0.1

sim = Sim(
	Epdf(0.0, 20.0, dx, num_particles, Z, ic),
	dt,
	b, k, c, brems, reset, rate)

paramstring = ""
paramstring *= b == 0.0 ? "" : @sprintf(" b-%.4f k-%.4f", b, k)
paramstring *= c == 0.0 ? "" : @sprintf(" c-%.4f", c)
paramstring *= Z == 1.0 ? "" : @sprintf(" Z-%.4f", Z)
paramstring *= !brems ?   "" : " brems"
paramstring *= !reset ?   "" : @sprintf(" reset rate-%4f", rate)

open("params.txt", "w") do io
	@printf(io, "N-1e%.2f dx-%g t-%g dt-%g ic-%s%s", log10(num_particles), dx, tf, dt, ic, paramstring)
end

@time trajectory(sim, tf)
