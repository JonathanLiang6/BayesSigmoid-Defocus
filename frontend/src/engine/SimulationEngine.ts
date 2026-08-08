import { LearningStep, PatientParams } from '../types';

export class SimulationEngine {
  private trueDose: number;
  private t: number;
  private patientParams: PatientParams;

  constructor(patientParams?: Partial<PatientParams>) {
    this.trueDose = 3.5 + Math.random() * 0.8;
    this.t = 0;
    
    this.patientParams = {
      age: patientParams?.age ?? 22,
      sex: patientParams?.sex ?? 1,
      axialLength: patientParams?.axialLength ?? 25.5,
      choroidalThickness: patientParams?.choroidalThickness ?? 10,
      cvi: patientParams?.cvi ?? 0.45,
      eyeSide: patientParams?.eyeSide ?? 0,
      rdv15: patientParams?.rdv15 ?? 0,
      refractionDefocus: patientParams?.refractionDefocus ?? 0,
    };

    const alFactor = (this.patientParams.axialLength - 24) * 0.12;
    const cviFactor = (this.patientParams.cvi - 0.45) * 0.4;
    this.trueDose = Math.max(3.5, Math.min(4.8, this.trueDose + alFactor + cviFactor));
  }

  nextRound(): LearningStep {
    this.t++;

    const decay = Math.exp(-this.t / 8);
    const oscillation = Math.sin(this.t * 0.8) * 1.0 * decay;
    const noise = (Math.random() - 0.5) * 0.3 * decay;
    const recommendedDose = this.trueDose + oscillation + noise;

    const uncertainty = 0.8 * decay + 0.05;

    const accuracy = Math.min(0.96, 0.6 + 0.36 * (1 - Math.exp(-this.t / 12)));
    const mae = 0.4 * Math.exp(-this.t / 10) + 0.02;

    const alWeight = 0.35 + noise * 0.05;
    const importance = {
      age: 0.3 + noise * 0.1,
      sex: 0.15,
      al: alWeight,
      choroidalThickness: 1 - (0.3 + noise * 0.1 + 0.15 + alWeight + 0.2),
      cvi: 0.2,
    };

    return {
      round: this.t,
      dose: recommendedDose,
      lowerBound: recommendedDose - uncertainty,
      upperBound: recommendedDose + uncertainty,
      accuracy,
      mae,
      importance,
    };
  }

  getTrueDose(): number {
    return this.trueDose;
  }

  reset(): void {
    this.t = 0;
    this.trueDose = 3.5 + Math.random() * 0.8;
  }

  shouldStop(currentUncertainty: number): boolean {
    return currentUncertainty < 0.05 || this.t >= 30;
  }
}