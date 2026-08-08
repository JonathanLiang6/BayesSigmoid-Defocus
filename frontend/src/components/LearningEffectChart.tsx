import { useEffect, useRef } from 'react';
import * as echarts from 'echarts';
import { LearningStep } from '../types';

interface Props {
  steps: LearningStep[];
}

export function LearningEffectChart({ steps }: Props) {
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
        right: '15%',
        top: '8%',
        bottom: '20%',
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
      yAxis: [
        {
          type: 'value',
          name: '准确率',
          nameLocation: 'middle',
          nameGap: 30,
          min: 0.5,
          max: 1.0,
          axisLine: { lineStyle: { color: '#3498db' } },
          axisLabel: {
            color: '#555',
            formatter: (value: number) => `${(value * 100).toFixed(0)}%`,
          },
          splitLine: { lineStyle: { type: 'dashed', color: '#e0e0e0' } },
        },
        {
          type: 'value',
          name: 'MAE',
          nameLocation: 'middle',
          nameGap: 30,
          min: 0,
          max: 0.5,
          axisLine: { lineStyle: { color: '#e74c3c' } },
          axisLabel: { color: '#555' },
          splitLine: { show: false },
        },
      ],
      series: [
        {
          name: '准确率',
          type: 'line',
          yAxisIndex: 0,
          smooth: true,
          symbol: 'circle',
          symbolSize: 8,
          lineStyle: { width: 3, color: '#3498db' },
          itemStyle: { color: '#3498db' },
          areaStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: 'rgba(52, 152, 219, 0.3)' },
              { offset: 1, color: 'rgba(52, 152, 219, 0.05)' },
            ]),
          },
          data: [],
          markLine: {
            silent: true,
            symbol: 'none',
            lineStyle: { type: 'dashed', width: 2, color: '#3498db' },
            label: { formatter: '临床阈值 90%', position: 'end' },
            data: [{ yAxis: 0.9 }],
          },
        },
        {
          name: 'MAE',
          type: 'line',
          yAxisIndex: 1,
          smooth: true,
          symbol: 'circle',
          symbolSize: 8,
          lineStyle: { width: 3, color: '#e74c3c' },
          itemStyle: { color: '#e74c3c' },
          data: [],
          markLine: {
            silent: true,
            symbol: 'none',
            lineStyle: { type: 'dashed', width: 2, color: '#e74c3c' },
            label: { formatter: '阈值', position: 'end' },
            data: [{ yAxis: 0.05 }],
          },
        },
      ],
      tooltip: {
        trigger: 'axis',
        formatter: (params: any) => {
          if (!params[0]) return '';
          const round = params[0].data[0];
          let html = `<strong>轮次 ${round}</strong><br/>`;
          params.forEach((p: any) => {
            if (p.seriesName === '准确率') {
              html += `${p.marker} 准确率: ${(p.data[1] * 100).toFixed(1)}%<br/>`;
            } else {
              html += `${p.marker} MAE: ${p.data[1].toFixed(4)}<br/>`;
            }
          });
          return html;
        },
      },
      legend: {
        data: ['准确率', 'MAE'],
        bottom: 0,
        textStyle: { color: '#555', fontSize: 11 },
        itemWidth: 20,
        itemHeight: 10,
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

    const accuracyData = steps.map((s) => [s.round, s.accuracy]);
    const maeData = steps.map((s) => [s.round, s.mae]);

    chartInstance.current.setOption({
      series: [
        { name: '准确率', data: accuracyData },
        { name: 'MAE', data: maeData },
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
        学习效果图 - 准确率与误差
      </h3>
      <p style={{ margin: '0 0 1rem 0', color: '#777', fontSize: '0.8rem' }}>
        双Y轴展示：准确率上升 (蓝) 与 MAE 下降 (红)
      </p>
      <div ref={chartRef} style={{ width: '100%', height: '320px' }} />
    </div>
  );
}