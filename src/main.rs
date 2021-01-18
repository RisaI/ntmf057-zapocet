use std::fs::File;
use std::io::Write;

const PI2: Fsize = 2.0 * std::f64::consts::PI as Fsize;
const DIM: usize = 2;
const EPSILON: Fsize = 1e-8;
const TIME_STEP: Fsize = 1e-4;
const PLOT_PERIOD: Fsize = TIME_STEP * 25.0;
const SIMULATION_STEPS: usize = (8.0 * 2.0 * std::f64::consts::PI / (TIME_STEP as f64)) as usize;

const COORD_ALIAS: [char; 10] = [ 'x', 'y', 'z', 'w', 'a', 'b', 'c', 'd', 'e', 'f' ];

#[cfg(not(feature = "earth-moon"))] const BINDING_CONST: Fsize = 1.0;
#[cfg(not(feature = "earth-moon"))] const CENTER_MASS: Fsize = 1.0;
#[cfg(not(feature = "earth-moon"))] const PARTICLES: usize = 1;
#[cfg(not(feature = "earth-moon"))] const MASSES: [Fsize; PARTICLES] = [ 1.0 ];

// Mass is expressed in masses of Earth
// Distance is expressed in 0.01AU
// Time is expressed in 1 year / 2 Pi
#[cfg(feature = "earth-moon")] const BINDING_CONST: Fsize = 3.00237029349437; // gravitational constants in our units
#[cfg(feature = "earth-moon")] const CENTER_MASS: Fsize = 333_054.253_181_514; // solar mass in masses of earth
#[cfg(feature = "earth-moon")] const PARTICLES: usize = 2;
#[cfg(feature = "earth-moon")] const MASSES: [Fsize; PARTICLES] = [ 1.0, 0.012_303_1469 ]; // mass of earth and mass of moon
#[cfg(feature = "earth-moon")] const EM_DIST: Fsize = 0.25_695_552_846_471; // distance of earth and moon in 0.01AU

type Fsize = f64;
type WorldState = [Fsize; 2 * DIM * PARTICLES];

fn add(a: &WorldState, b: &WorldState) -> WorldState {
    let mut res: WorldState = a.clone();

    for i in 0..(2 * DIM * PARTICLES) {
        res[i] += b[i];
    }

    res
}

fn smul(scalar: Fsize, a: &WorldState) -> WorldState {
    let mut res: WorldState = a.clone();

    for i in 0..(2 * DIM * PARTICLES) {
        res[i] *= scalar;
    }

    res
}

fn dist(a: &[Fsize], b: &[Fsize]) -> Fsize {
    let mut psum: Fsize = 0.0;

    for d in 0..DIM {
        let dist = a[d] - b[d];
        psum += dist * dist;
    }

    psum.sqrt()
}

// Central field potential for non-interacting particles
fn potential(state: &WorldState) -> Fsize {
    let mut sum: Fsize = 0.0;

    for i in 0..PARTICLES {
        sum += BINDING_CONST * CENTER_MASS * MASSES[i] / dist(&state[(2 * DIM * i)..((2 * i + 1) * DIM)], &[0.0; DIM]);

        for j in (i + 1)..PARTICLES {
            sum += BINDING_CONST * MASSES[i] * MASSES[j] / dist(&state[(2 * DIM * i)..((2 * i + 1) * DIM)], &state[(2 * DIM * j)..((2 * j + 1) * DIM)])
        }
    }
    
    -sum
}

fn hamiltonian(state: &WorldState) -> Fsize {
    let mut kin: Fsize = 0.0;

    for i in 0..PARTICLES {
        for j in 0..DIM {
            let val = state[2 + j + i * 2 * DIM];
            kin += val * val / MASSES[i];
        }
    }

    0.5 * kin + potential(&state)
}

fn angular_momentum(state: &WorldState) -> Fsize {
    let mut ang: Fsize = 0.0;

    for i in 0..PARTICLES {
        let p = &state[(i * 2 * DIM)..((i + 1) * 2 * DIM)];
        ang += p[0] * p[DIM + 1] - p[1] * p[DIM];
    }

    ang
}

fn pd_state_fn(fnc: fn(&WorldState) -> Fsize, state: &WorldState, arg: usize) -> Fsize {
    let mut sp = state.clone();
    let mut sm = state.clone();

    sp[arg] = sp[arg] + EPSILON;
    sm[arg] = sm[arg] - EPSILON;

    (fnc(&sp) - fnc(&sm)) / (2.0 * EPSILON)
}

fn rhs(state: &WorldState) -> WorldState {
    let mut rhs: WorldState = [0.0; 2 * DIM * PARTICLES];

    for i in 0..PARTICLES {
        let start = i * 2 * DIM; // First index of particle state

        for j in 0..DIM {
            rhs[start + j] = pd_state_fn(hamiltonian, state, start + j + DIM);
            rhs[start + j + DIM] = -pd_state_fn(hamiltonian, state, start + j);
        }
    }
    
    rhs
}

#[cfg(not(feature = "earth-moon"))]
fn initial_conditions() -> WorldState {
    const EPSILONS: [Fsize; PARTICLES] = [ 0.5 ];
    let mut state: WorldState = [0.0; PARTICLES * 2 * DIM];

    for i in 0..PARTICLES {
        let epsilon = EPSILONS[i];
        let start = 2*DIM * i;

        state[start] = 1.0 - epsilon;
        state[start + 2 * DIM - 1] = MASSES[i] * ( (1.0 + epsilon) / (1.0 - epsilon) ).sqrt();
    }
    
    state
}

#[cfg(feature = "earth-moon")]
fn initial_conditions() -> WorldState {
    [
        101.671,           0.0, 0.0, MASSES[0] * 98.2699414604534, // aphelion distance of earth, momentum in aphelion
        101.671 + EM_DIST, 0.0, 0.0, MASSES[1] * 100.707191536756  // Moon
    ]
}

// second order Runge-Kutta, Butcher tableau
//  0  |
// 1/2 | 1/2 
// ----|-------
//     |  0  1
fn rk2(state: &WorldState, time_step: Fsize) -> WorldState {
    let k = rhs(&add(state, &smul(time_step * 0.5, &rhs(state))));
    smul(time_step, &k)
}

// fourth order Runge-Kutta, Butcher tableau
//  0  |
// 1/2 | 1/2 
// 1/2 |  0  1/2 
//  1  |  0   0   1
// ----|----------------
//     | 1/6 1/3 1/3 1/6
fn rk4(state: &WorldState, time_step: Fsize) -> WorldState {
    const A: [Fsize; 4] = [ 0.0, 0.5, 0.5, 1.0 ];
    let mut k: [WorldState; 4] = [[0.0; 2 * DIM * PARTICLES]; 4];

    k[0] = rhs(state);
    for i in 1..4 {
        k[i] = rhs(&add(state, &smul(time_step * A[i], &k[i - 1])));
    }

    const B: [Fsize; 4] = [ 1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0 ];
    let mut res: WorldState = smul(B[0], &k[0]);

    for i in 1..4 {
        res = add(&res, &smul(B[i], &k[i]));
    }
    
    smul(time_step, &res)
}

fn write_header(file: &mut File) -> std::io::Result<()> {
    file.write_all(b"# ")?; 
    for i in 0..PARTICLES {
        for j in 0..DIM { file.write_all(format!("\t{}_{}",  COORD_ALIAS[j], i).as_bytes())?; }
        for j in 0..DIM { file.write_all(format!("\tp{}_{}", COORD_ALIAS[j], i).as_bytes())?; }
    }
    file.write_all(b"\tH\tL\n")?;
    Ok(())
}

fn write_state(file: &mut File, time: Fsize, state: &WorldState) -> std::io::Result<()> {
    file.write_all(format!("{:.3}", time).as_bytes())?;
    for i in 0..(2 * DIM * PARTICLES) {
        file.write_all(format!("\t{:.7}", state[i]).as_bytes())?;
    }
    file.write_all(format!("\t{:.8}\t{:.8}\n", hamiltonian(&state), angular_momentum(&state)).as_bytes())?;
    Ok(())
}

fn main() -> std::io::Result<()> {
    let mut state: WorldState = initial_conditions();
    let mut time: Fsize = 0.0;

    let mut integrator: fn(&WorldState, Fsize) -> WorldState = rk4;
    let mut integrator_name = "rk4";
    let mut out: File;

    // Program arguments
    {
        let mut out_file: Option<String> = Option::None;

        let args: Vec<String> = std::env::args().collect();
        for i in 1..args.len()
        {
            if &args[i] == "--rk2" {
                println!("Downgrading to second-order Runge-Kutta.");
                integrator = rk2;
                integrator_name = "rk2";
            } else if &args[i] == "-f" && i < args.len() - 1 {
                out_file = Option::Some(args[i + 1].clone());
            }
        }

        out = File::create(
            if let Option::Some(val) = out_file {
                val
            } else if cfg!(feature = "earth-moon") {
                String::from("result-em.txt")
            } else {
                String::from("result.txt")
            }
        )?;
    }

    write_header(&mut out)?; // Write file header
    write_state(&mut out, time, &state)?; // Write initial state

    let mut plot_time: Fsize = 0.0;
    let mut orig_pos: [Fsize; DIM] = [ 0.0; DIM ];
    orig_pos.copy_from_slice(&state[0..DIM]);
    println!("Simulating {} steps.", SIMULATION_STEPS);

    for _i in 0..SIMULATION_STEPS {
        state = add(&state, &integrator(&state, TIME_STEP));

        if time % PI2 > (time + TIME_STEP) % PI2 {
            print!("Complete revolution at {:.4}, distance from original position = {:.8}.\n", time, dist(&orig_pos, &state[0..DIM]));
        }

        time += TIME_STEP;
        plot_time += TIME_STEP;

        if plot_time >= PLOT_PERIOD {
            write_state(&mut out, time, &state)?;
            plot_time = 0.0;
        }
    }

    
    if cfg!(not(feature = "earth-moon")) {
        println!("Benchmarking integrator...");
        out = File::create(format!("benchmark-{}.txt", integrator_name))?;

        for i in 1..1_001 {
            state = initial_conditions();
            time = 0.0;

            let time_step: Fsize = (i as Fsize) * 1e-4;
            let steps = (2.0 * std::f64::consts::PI / (time_step as f64)) as usize; 

            for _j in 0..steps {
                state = add(&state, &integrator(&state, time_step));
                time += time_step;
            }

            state = add(&state, &integrator(&state, (2.0 * std::f64::consts::PI - (time as f64)) as Fsize));

            let dst = dist(&orig_pos, &state[0..DIM]);
            let line = format!("{:.7}\t{}\n", time_step, dst);

            out.write_all(line.as_bytes())?;
        }
    }

    Ok(())
}
