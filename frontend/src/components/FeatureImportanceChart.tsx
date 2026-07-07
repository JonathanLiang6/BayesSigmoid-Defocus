import { useEffect, useRef } from 'react';
import * as echarts from 'echarts';
import { LearningStep } from '../types';

interface Props {
  steps: LearningStep[];
}

export function FeatureImportanceChart({ steps }: Props) {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.ECharts | null>(null);

  const featureNames: Record<string, string> = {
    age: '年龄',
    sex: '性别',
    al: '眼轴长度',
    choroidalThickness: '脉络膜厚度',
    cvi: 'CVI',
  };

  useEffect(() => {
    if (!chartRef.current) return;

    chartInstance.current = echarts.init(chartRef.current);

    const option: echarts.EChartsOption = {
      animation: true,
      animationDuration: 300,
      backgroundColor: 'transparent',
      grid: {
        left: '15%',
        right: '5%',
        top: '10%',
        bottom: '15%',
      },
      xAxis: {
        type: 'value',
        name: '贡献度',
        nameLocation: 'middle',
        nameGap: 30,
        min: 0,
        max: 0.5,
        axisLine: { lineStyle: { color: '#555' } },
        axisLabel: { color: '#555' },
        splitLine: { lineStyle: { type: 'dashed', color: '#e0e0e0' } },
      },
      yAxis: {
        type: 'category',
        data: ['年龄', '性别', '眼轴长度', '脉络膜厚度', 'CVI'],
        axisLine: { lineStyle: { color: '#555' } },
        axisLabel: { color: '#555' },
      },
      series: [
        {
          name: '特征贡献度',
          type: 'bar',
          barWidth: '60%',
          itemStyle: {
            borderRadius: [0, 4, 4, 0],
            color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [
              { offset: 0, color: '#3498db' },
              { offset: 1, color: '#2ecc71' },
            ]),
          },
          data: [],
          label: {
            show: true,
            position: 'right',
            formatter: '{c}',
            color: '#555',
          },
        },
      ],
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        formatter: (params: any) => {
          if (!params[0]) return '';
          const p = params[0];
          return `${p.name}: ${(p.value * 100).toFixed(1)}%`;
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

    const latestImportance = steps[steps.length - 1].importance;
    const data = Object.entries(latestImportance)
      .map(([key, value]) => ({
        name: featureNames[key] || key,
        value: Math.max(0, Math.min(0.5, value)),
      }))
      .sort((a, b) => b.value - a.value);

    chartInstance.current.setOption({
      yAxis: {
        data: data.map((d) => d.name),
      },
      series: [
        {
          name: '特征贡献度',
          data: data.map((d) => d.value),
        },
      ],
    });
  }, [steps]);

  return (
    <div style={{
      background: 'white',
      borderRadius: '1rem',
      padding: '1rem',
      boxShadow: '0 2px 12px rgba(0, 0, 0, 0.08)',
      height: '100%',
    }}>
      <h3 style={{ margin: '0 0 0.5rem 0', color: '#1e3a5f', fontSize: '1rem' }}>
        特征贡献度分析
      </h3>
      <p style={{ margin: '0 0 1rem 0', color: '#777', fontSize: '0.8rem' }}>
        各特征对剂量调整的影响权重
      </p>
      <div ref={chartRef} style={{ width: '100%', height: '280px' }} />
    </div>
  );
}