export interface PatientParams {
  age: number;
  sex: number;
  axialLength: number;
  choroidalThickness: number;
  cvi: number;
  eyeSide: number;
  rdv15: number;
  refractionDefocus: number;
}

export interface PatientInfo {
  id: string;
  subject_id: number;
  eye: string;
  batch: string;
  age: number;
  gender: number;
  axial_length: number;
  choroid_thickness: number;
  cvi: number;
  rdv15: number;
  refractive_error: number;
  dose: number;
}

export interface LearningStep {
  round: number;
  dose: number;
  lowerBound: number;
  upperBound: number;
  accuracy: number;
  mae: number;
  importance: Record<string, number>;
}

export interface LearningResult {
  session_id: string;
  patient_id: string;
  iterations: number[];
  uncertainties: number[];
  recommended_doses: number[];
  predicted_responses: number[];
  dose_response_curves: Array<{ dose: number[]; response: number[] }>;
  final_dose: number;
  final_response: number;
  final_uncertainty: number;
  confidence: number;
  converged: boolean;
}

export interface SimulationState {
  isRunning: boolean;
  isPaused: boolean;
  currentRound: number;
  maxRounds: number;
  speed: number;
  steps: LearningStep[];
  finalDose: number | null;
  converged: boolean;
}

export interface ChartDataPoint {
  round: number;
  dose: number;
  lowerBound: number;
  upperBound: number;
  uncertainty: number;
}