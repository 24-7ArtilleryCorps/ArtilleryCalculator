// Constants
const g = 9.81;  // Acceleration due to gravity (m/s^2)

// Convert degrees to mils
function degreesToMils(degrees) {
    return degrees * 17.7777777778;
}

// Function to simulate projectile trajectory
function simulateTrajectory(angleDeg, targetRange, heightLauncher, heightTarget,
                            projectileMass = 23, projectileAirDrag = 0.0043, projectileVelocity = 212.5) {
    const angleRad = angleDeg * (Math.PI / 180);
    let vx0 = projectileVelocity * Math.cos(angleRad);
    let vy0 = projectileVelocity * Math.sin(angleRad);
    let x = 0;
    let y = heightLauncher;
    const dt = 0.01;
    const tMax = 100;

    let timeOfFlight = 0;

    for (let step = 0; step < tMax / dt; step++) {
        timeOfFlight += dt;

        const v = Math.sqrt(vx0 * vx0 + vy0 * vy0);
        const dragForce = (v !== 0) ? projectileAirDrag * v * v : 0;
        const ax = -dragForce * vx0 / (projectileMass * v);
        const ay = -g - (dragForce * vy0 / (projectileMass * v));

        vx0 += ax * dt;
        vy0 += ay * dt;
        x += vx0 * dt;
        y += vy0 * dt;

        if (x >= targetRange && y <= heightTarget) {
            break;
        }
    }

    return { x, y, timeOfFlight };
}

// Objective function for optimization
function objectiveFunction(angleDeg, targetRange, heightLauncher, heightTarget,
                           projectileMass, projectileAirDrag, projectileVelocity) {
    const { x: distanceTraveled, y: finalHeight } = simulateTrajectory(
        angleDeg, targetRange, heightLauncher, heightTarget,
        projectileMass, projectileAirDrag, projectileVelocity);

    return Math.abs(distanceTraveled - targetRange) + Math.abs(finalHeight - heightTarget);
}

// Function to calculate the range and bearing
function rangeCalculation(eastingLauncher, northingLauncher, heightLauncher,
                          eastingTarget, northingTarget, heightTarget) {
    const deltaEasting = eastingTarget - eastingLauncher;
    const deltaNorthing = northingTarget - northingLauncher;
    const horizontalDistance = Math.sqrt(deltaEasting * deltaEasting + deltaNorthing * deltaNorthing);

    const bearingRad = Math.atan2(deltaEasting, deltaNorthing);
    let bearingDeg = bearingRad * (180 / Math.PI);
    if (bearingDeg < 0) {
        bearingDeg += 360;
    }

    return { horizontalDistance, bearingDeg, heightLauncher, heightTarget };
}

// Golden-section search for optimal angle
function goldenSectionSearch(lowerBound, upperBound, objectiveFunc, tol = 0.001) {
    const phi = (1 + Math.sqrt(5)) / 2;
    let c = upperBound - (upperBound - lowerBound) / phi;
    let d = lowerBound + (upperBound - lowerBound) / phi;

    while (Math.abs(c - d) > tol) {
        if (objectiveFunc(c) < objectiveFunc(d)) {
            upperBound = d;
        } else {
            lowerBound = c;
        }
        c = upperBound - (upperBound - lowerBound) / phi;
        d = lowerBound + (upperBound - lowerBound) / phi;
    }

    return (upperBound + lowerBound) / 2;
}

// Main function
function main(eastingLauncher, northingLauncher, heightLauncher,
              eastingTarget, northingTarget, heightTarget,
              projectileMass = 23, projectileAirDrag = 0.0043, projectileVelocity = 212.5) {

    const { horizontalDistance, bearingDeg } = rangeCalculation(
        eastingLauncher, northingLauncher, heightLauncher,
        eastingTarget, northingTarget, heightTarget);

    // Golden-section search to find the optimal launch angle below 45 degrees
    const optimalAngleDegBelow = goldenSectionSearch(0, 45, angleDeg => 
        objectiveFunction(angleDeg, horizontalDistance, heightLauncher, heightTarget, projectileMass, projectileAirDrag, projectileVelocity));
    const milsDirect = Math.round(degreesToMils(optimalAngleDegBelow));
    const { timeOfFlight: tofDirect } = simulateTrajectory(optimalAngleDegBelow, horizontalDistance, heightLauncher, heightTarget,
                                                           projectileMass, projectileAirDrag, projectileVelocity);

    // Golden-section search to find the optimal launch angle above 45 degrees
    const optimalAngleDegAbove = goldenSectionSearch(45, 90, angleDeg =>
        objectiveFunction(angleDeg, horizontalDistance, heightLauncher, heightTarget, projectileMass, projectileAirDrag, projectileVelocity));
    const milsIndirect = Math.round(degreesToMils(optimalAngleDegAbove) - 17);
    const { timeOfFlight: tofIndirect } = simulateTrajectory(optimalAngleDegAbove, horizontalDistance, heightLauncher, heightTarget,
                                                             projectileMass, projectileAirDrag, projectileVelocity);

    // Output values
    return {
        horizontalDistance: horizontalDistance,
        bearingDeg: bearingDeg.toFixed(2),
        milsIndirect: milsIndirect,
        tofIndirect: tofIndirect.toFixed(2),
        milsDirect: milsDirect,
        tofDirect: tofDirect.toFixed(2)
    };
}

function observerGridCalc(northingLauncher, eastingLauncher, heightLauncher,
                          eastingObserver, northingObserver, observerBearingToTarget, observerRangeToTarget, heightTarget) {
    
    // Convert bearing to radians
    const bearingRad = observerBearingToTarget * (Math.PI / 180);

    // Calculate the target's easting and northing using the observer's bearing and range to the target
    const eastingTarget = eastingObserver + observerRangeToTarget * Math.sin(bearingRad);
    const northingTarget = northingObserver + observerRangeToTarget * Math.cos(bearingRad);

    // Call main to finalize the calculation
    const result = main(eastingLauncher, northingLauncher, heightLauncher, eastingTarget, northingTarget, heightTarget);

    // Return an object including the main result, eastingTarget, and northingTarget
    return {
        result: result,               // The output from the main function
        updatedeastingTarget: eastingTarget, // Calculated eastingTarget
        updatednorthingTarget: northingTarget // Calculated northingTarget
    };
}

// Example usage
const result = observerGridCalc(1000, 1000, 10, 7596, 4299, 90, 1000, 120);
console.log(result);
