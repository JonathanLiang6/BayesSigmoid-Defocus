import { useState, useRef, useCallback, useEffect } from 'react';
import { LearningStep, LearningResult } from '../types';

const API_BASE = 'http://localhost:8000';

export function useSimulation(maxRounds: number = 30) {
  const [isRunning, setIsRunning] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [currentRound, setCurrentRound] = useState(0);
  const [speed, setSpeed] = useState(1);
  const [steps, setSteps] = useState<LearningStep[]>([]);
  const [finalDose, setFinalDose] = useState<number | null>(null);
  const [converged, setConverged] = useState(false);
  const [learningResult, setLearningResult] = useState<LearningResult | null>(null);

  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const allStepsRef = useRef<LearningStep[]>([]);

  const clearTimer = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const runStep = useCallback(() => {
    setCurrentRound((prev) => {
      const next = prev + 1;
      if (next >= allStepsRef.current.length) {
        clearTimer();
        setIsRunning(false);
        setConverged(true);
        return allStepsRef.current.length;
      }
      setSteps(allStepsRef.current.slice(0, next));
      return next;
    });
  }, [clearTimer]);

  const startTimer = useCallback(() => {
    clearTimer();
    timerRef.current = setInterval(runStep, 500 / speed);
  }, [speed, runStep, clearTimer]);

  const convertToSteps = useCallback((result: LearningResult): LearningStep[] => {
    const { iterations, uncertainties, recommended_doses, predicted_responses } = result;
    
    return iterations.map((round, i) => {
      const uncertainty = uncertainties[i];
      const dose = recommended_doses[i];
      const response = predicted_responses[i];
      
      const mae = Math.max(0.01, uncertainty * 0.8);
      const accuracy = Math.min(0.98, 0.6 + (1 - uncertainty) * 0.38);
      
      return {
        round,
        dose,
        lowerBound: dose - uncertainty,
        upperBound: dose + uncertainty,
        accuracy,
        mae,
        importance: {
          '脉络膜厚度': 0.35 - i * 0.01,
          'CVI': 0.25 - i * 0.005,
          '眼轴长度': 0.2 - i * 0.003,
          '年龄': 0.15 - i * 0.002,
          '性别': 0.05,
        },
      };
    });
  }, []);

  const start = useCallback(async (patientId: string) => {
    clearTimer();
    setIsLoading(true);
    setIsRunning(false);
    setIsPaused(false);
    setCurrentRound(0);
    setSteps([]);
    setFinalDose(null);
    setConverged(false);
    setLearningResult(null);
    allStepsRef.current = [];

    try {
      const response = await fetch(`${API_BASE}/api/learning/run/${patientId}`, {
        method: 'POST',
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      
      const result: LearningResult = await response.json();
      setLearningResult(result);
      setFinalDose(result.final_dose);
      
      const convertedSteps = convertToSteps(result);
      allStepsRef.current = convertedSteps;
      
      setIsLoading(false);
      setIsRunning(true);
      startTimer();
    } catch (error) {
      console.error('Failed to run learning:', error);
      setIsLoading(false);
    }
  }, [convertToSteps, startTimer, clearTimer]);

  const pause = useCallback(() => {
    clearTimer();
    setIsPaused(true);
    setIsRunning(false);
  }, [clearTimer]);

  const resume = useCallback(() => {
    if (currentRound < allStepsRef.current.length) {
      setIsPaused(false);
      setIsRunning(true);
      startTimer();
    }
  }, [currentRound, startTimer]);

  const reset = useCallback(() => {
    clearTimer();
    setIsRunning(false);
    setIsPaused(false);
    setIsLoading(false);
    setCurrentRound(0);
    setSteps([]);
    setFinalDose(null);
    setConverged(false);
    setLearningResult(null);
    allStepsRef.current = [];
  }, [clearTimer]);

  const seek = useCallback((round: number) => {
    const targetRound = Math.max(0, Math.min(round, allStepsRef.current.length));
    setCurrentRound(targetRound);
    setSteps(allStepsRef.current.slice(0, targetRound));
    
    if (targetRound >= allStepsRef.current.length) {
      clearTimer();
      setIsRunning(false);
      setConverged(true);
    }
  }, [clearTimer]);

  const setSpeedHandler = useCallback((newSpeed: number) => {
    setSpeed(newSpeed);
    if (isRunning && !isPaused) {
      clearTimer();
      timerRef.current = setInterval(runStep, 500 / newSpeed);
    }
  }, [isRunning, isPaused, runStep, clearTimer]);

  useEffect(() => {
    return () => clearTimer();
  }, [clearTimer]);

  return {
    isRunning,
    isPaused,
    isLoading,
    currentRound,
    maxRounds: allStepsRef.current.length || maxRounds,
    speed,
    steps,
    finalDose,
    converged,
    learningResult,
    start,
    pause,
    resume,
    reset,
    setSpeed: setSpeedHandler,
    seek,
  };
}