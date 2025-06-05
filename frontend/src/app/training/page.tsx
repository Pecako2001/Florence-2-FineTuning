"use client";
import { useState } from 'react';
import { Button, NumberInput, Select, Progress, Stack, Slider } from '@mantine/core';
import Navbar from '../../components/Navbar';

export default function TrainingPage() {
  const [batch, setBatch] = useState(8);
  const [epochs, setEpochs] = useState(1);
  const [split, setSplit] = useState(0.2);
  const [task, setTask] = useState('VQA');
  const [progress, setProgress] = useState(0);

  const startTraining = () => {
    setProgress(10);
    fetch('/api/train', {
      method: 'POST',
      body: JSON.stringify({ batch, epochs, split, task }),
    })
      .then(() => {
        let val = 10;
        const interval = setInterval(() => {
          val += 10;
          setProgress(val);
          if (val >= 100) clearInterval(interval);
        }, 500);
      })
      .catch(() => setProgress(0));
  };

  return (
    <>
      <Navbar />
      <main className="p-4 max-w-md mx-auto">
        <Stack>
          <Select
            label="Task"
            data={[{ value: 'VQA', label: 'VQA' }, { value: 'Object Detection', label: 'Object Detection' }]}
            value={task}
            onChange={(v) => setTask(v || 'VQA')}
          />
          <NumberInput label="Batch Size" value={batch} onChange={(v) => setBatch(Number(v))} />
          <NumberInput label="Epochs" value={epochs} onChange={(v) => setEpochs(Number(v))} />
          <Slider label="Validation Split" value={split} onChange={setSplit} min={0} max={1} step={0.1} />
          <Button onClick={startTraining}>Start Training</Button>
          {progress > 0 && <Progress value={progress} animated />}
        </Stack>
      </main>
    </>
  );
}
