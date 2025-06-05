"use client";
import { useState } from 'react';
import { Button, FileInput, Select, TextInput, Stack } from '@mantine/core';
import Navbar from '../../components/Navbar';

export default function EvaluationPage() {
  const [file, setFile] = useState<File | null>(null);
  const [prompt, setPrompt] = useState('');
  const [task, setTask] = useState('VQA');
  const [output, setOutput] = useState<string | null>(null);

  const handleEval = () => {
    if (!file) return;
    const form = new FormData();
    form.append('image', file);
    form.append('prompt', prompt);
    form.append('task', task);

    fetch('/api/evaluate', { method: 'POST', body: form })
      .then((r) => r.text())
      .then(setOutput);
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
          <FileInput label="Upload Image" onChange={setFile} />
          <TextInput label="Prompt" value={prompt} onChange={(e) => setPrompt(e.currentTarget.value)} />
          <Button onClick={handleEval}>Evaluate</Button>
          {output && (
            <div>
              <strong>Output:</strong>
              <p>{output}</p>
            </div>
          )}
        </Stack>
      </main>
    </>
  );
}
