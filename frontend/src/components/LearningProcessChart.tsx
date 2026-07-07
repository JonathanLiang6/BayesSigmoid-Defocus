import { useEffect, useRef } from 'react';
import * as echarts from 'echarts';
import { LearningStep } from '../types';

interface Props {
  steps: LearningStep[];
  converged: boolean;
  finalDose: number | null;
}

export function LearningProcessChart({ steps, converged, finalDose }: Props) {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.ECharts | null>(null);

  useEffect(() => {
    if (!chartRef.current) return;

    chartInstance.current = echarts.init(chartRef.current);

    const option: echarts.EChartsOption = {
      animation: true,
      animationDuration: 300,
      animationEasing: 'cubicInOut',
      backgroundColor: 'transparent',
      grid: {
        left: '10%',
        right: '10%',
        top: '12%',
        bottom: '15%',
      },
      xAxis: {
        type: 'value',
        name: '学习轮次',
        nameLocation: 'middle',
        nameGap: 30,
        min: 0,
        max: 20,
        splitLine: {
          lineStyle: { type: 'dashed', color: '#e0e0e0' },
        },
        axisLine: { lineStyle: { color: '#555' } },
        axisLabel: { color: '#555' },
      },
      yAxis: {
        type: 'value',
        name: '推荐离焦剂量 (D)',
        nameLocation: 'middle',
        nameGap: 45,
        min: 3,
        max: 5.5,
        splitLine: {
          lineStyle: { type: 'dashed', color: '#e0e0e0' },
        },
        axisLine: { lineStyle: { color: '#555' } },
        axisLabel: { color: '#555' },
      },
      series: [
        {
          name: '95% HDI区间',
          type: 'custom',
          renderItem: (_params, api) => {
            if (api.value(0) === undefined) return null;
            const x = api.coord([api.value(0), 0])[0];
            const y0 = api.coord([0, api.value(1)])[1];
            const y1 = api.coord([0, api.value(2)])[1];
            return {
              type: 'rect',
              shape: { x, y: y0, width: 20, height: y1 - y0 },
              style: {
                fill: 'rgba(45, 90, 135, 0.15)',
              },
            };
          },
          data: [],
          z: 1,
        },
        {
          name: '推荐剂量',
          type: 'line',
          smooth: true,
          symbol: 'circle',
          symbolSize: 10,
          lineStyle: { width: 3, color: '#2d5a87' },
          itemStyle: { color: '#2d5a87' },
          data: [],
          z: 3,
        },
        {
          name: '收敛线',
          type: 'line',
          markLine: {
            silent: true,
            symbol: 'none',
            lineStyle: { type: 'dashed', width: 2, color: '#27ae60' },
            data: [],
          },
          z: 4,
        },
      ],
      tooltip: {
        trigger: 'axis',
        formatter: (params: any) => {
          if (!params[0]) return '';
          const step = params[0];
          return `轮次: ${step.data[0]}<br/>剂量: ${step.data[1].toFixed(3)} D`;
        },
      },
    };

    chartInstance.current.setOption(option);

    const handleResize = () => chartInstance.current?.resize();
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chartInstance.current?.dispose();
    };
  }, []);

  useEffect(() => {
    if (!chartInstance.current || steps.length === 0) return;

    const doses = steps.map((s) => [s.round, s.dose]);
    const bounds = steps.map((s) => [s.round, s.lowerBound, s.upperBound]);

    chartInstance.current.setOption({
      series: [
        {
          name: '95% HDI区间',
          data: bounds,
        },
        {
          name: '推荐剂量',
          data: doses,
        },
        {
          name: '收敛线',
          markLine: {
            data: finalDose !== null ? [{ yAxis: finalDose }] : [],
          },
        },
      ],
    });
  }, [steps, finalDose]);

  return (
    <div style={{
      background: 'white',
      borderRadius: '1rem',
      padding: '1rem',
      boxShadow: '0 2px 12px rgba(0, 0, 0, 0.08)',
      height: '100%',
    }}>
      <h3 style={{ margin: '0 0 0.5rem 0', color: '#1e3a5f', fontSize: '1rem' }}>
        学习过程图 - 震荡收敛
      </h3>
      <p style={{ margin: '0 0 1rem 0', color: '#777', fontSize: '0.8rem' }}>
        观察推荐剂量随迭代逐渐收敛至最优值
      </p>
      <div ref={chartRef} style={{ width: '100%', height: '300px' }} />
      {converged && finalDose !== null && (
        <div style={{
          textAlign: 'center',
          marginTop: '1rem',
          padding: '0.8rem',
          background: '#e8f8f0',
          borderRadius: '0.5rem',
          color: '#27ae60',
          fontWeight: 600,
        }}>
          ✓ 已收敛 | 最终推荐剂量: {finalDose.toFixed(2)} D
        </div>
      )}
    </div>
  );
}