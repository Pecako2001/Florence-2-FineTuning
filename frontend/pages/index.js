import { useState } from 'react';

export default function Home() {
  const [image, setImage] = useState(null);
  const [annotation, setAnnotation] = useState(null);
  const [task, setTask] = useState('DocVQA');
  const [status, setStatus] = useState('');

  const upload = async () => {
    if (!image || !annotation) return;
    const form = new FormData();
    form.append('image', image);
    form.append('annotation', annotation);
    await fetch('http://localhost:8000/upload', { method: 'POST', body: form });
    setStatus('uploaded');
  };

  const train = async () => {
    const res = await fetch('http://localhost:8000/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ epochs: 1, batch_size: 1, task_type: task })
    });
    const data = await res.json();
    setStatus('training pid ' + data.pid);
  };

  return (
    <div className="p-4">
      <h1 className="text-xl font-bold">Florence-2 Fine-tuning</h1>
      <div>
        <input type="file" onChange={e => setImage(e.target.files[0])} />
        <input type="file" onChange={e => setAnnotation(e.target.files[0])} />
        <select onChange={e => setTask(e.target.value)} value={task}>
          <option value="DocVQA">VQA</option>
          <option value="ObjectDetection">Object Detection</option>
        </select>
        <button onClick={upload}>Upload</button>
        <button onClick={train}>Train</button>
      </div>
      <div>{status}</div>
    </div>
  );
}
