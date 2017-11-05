use std::f64::INFINITY;

// All functions translated into rust from John D Cooks's C code:
// https://www.johndcook.com/Gamma.cpp
const HALF_LOG_2PI: f64 = 0.91893853320467274178032973640562;
const GAMMA: f64 = 0.577215664901532860606512090;  // Euler's gamma constant
const GAMMA_NUMER: [f64; 8] = [-1.71618513886549492533811E+0,
                                2.47656508055759199108314E+1,
                               -3.79804256470945635097577E+2,
                                6.29331155312818442661052E+2,
                                8.66966202790413211295064E+2,
                               -3.14512729688483675254357E+4,
                               -3.61444134186911729807069E+4,
                                6.64561438202405440627855E+4];
const GAMMA_DENOM: [f64; 8] =  [-3.08402300119738975254353E+1,
                                 3.15350626979604161529144E+2,
                                -1.01515636749021914166146E+3,
                                -3.10777167157231109440444E+3,
                                 2.25381184209801510330112E+4,
                                 4.75584627752788110767815E+3,
                                -1.34659959864969306392456E+5,
                                -1.15132259675553483497211E+5];
const GAMMALN_C: [f64; 8] = [ 1.0/12.0,
                             -1.0/360.0,
                             1.0/1260.0,
                             -1.0/1680.0,
                             1.0/1188.0,
                             -691.0/360360.0,
                             1.0/156.0,
                             -3617.0/122400.0];

pub fn gamma(z: f64) -> f64 {
    if z <= 0.0 {
        panic!("Cannot compute gamma on negative value");
    }

    if z < 0.001 {
        1.0/(z * (1.0 + GAMMA * z))

    } else if z < 12.0 {

        let mut y = z;
        let mut n: usize = 0;
        let arg_was_less_than_one = y < 1.0;

        // Add or subtract integers as necessary to bring y into (1,2)
        // Will correct for this below
        if arg_was_less_than_one {
            y += 1.0;
        } else {
            n = (y as usize) - 1;  // will use n later
            y -= n as f64;
        }

        assert!( 1.0 <= y && y <= 2.0);

        let mut numer = 0.0;
        let mut denom = 1.0;

        let x = y - 1.0;
        for i in 0..8 {
            numer = (numer + GAMMA_NUMER[i]) * x;
            denom = denom * x + GAMMA_DENOM[i];
        }
        let mut result = numer/denom + 1.0;

        // Apply correction if argument was not initially in (1,2)
        if arg_was_less_than_one {
            // Use identity gamma(z) = gamma(z+1)/z
            // The variable "result" now holds gamma of the original y + 1
            // Thus we use y-1 to get back the orginal y.
            result /= y - 1.0;
        } else {
            // Use the identity gamma(z+n) = z*(z+1)* ... *(z+n-1)*gamma(z)
            for _ in 0..n {
                result *= y;
                y += 1.0;
            }
        }

		result
    } else if z > 171.624 {
        // too big
        INFINITY        
    } else {
        gammaln(z).exp()
    }
}


pub fn gammaln(z: f64) -> f64 {
    if z <= 0.0 {
        panic!("Cannot compute gammaln on negative value");
    }

    if z < 12.0 {
        gamma(z).ln()
    } else {
        let x = 1.0/(z*z);
        let mut sum = GAMMALN_C[7];
        for i in (0..7).rev() {
            sum *= x;
            sum += GAMMALN_C[i];
        }
        let series = sum/z;

        (z - 0.5) * z.ln() - z + HALF_LOG_2PI + series
    }
}

