"use client";
import { useState, useEffect } from "react";
import { Grid, Button, Modal, FileInput, Group } from "@mantine/core";
import Navbar from "../../components/Navbar";
import DatasetCard from "../../components/DatasetCard";

export default function DatasetPage() {
  const [datasets, setDatasets] = useState<string[]>([]);
  const [opened, setOpened] = useState(false);
  const [file, setFile] = useState<File | null>(null);

  useEffect(() => {
    fetch("/api/datasets")
      .then((r) => r.json())
      .then(setDatasets)
      .catch(() => {});
  }, []);

  const handleUpload = () => {
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    fetch("/api/datasets", { method: "POST", body: form }).then(() => {
      setOpened(false);
      setFile(null);
      setDatasets((d) => [...d, file.name]);
    });
  };

  const handleDelete = (name: string) => {
    fetch(`/api/datasets/${name}`, { method: "DELETE" }).then(() =>
      setDatasets((d) => d.filter((n) => n !== name))
    );
  };

  return (
    <>
      <Navbar />
      <main className="p-4">
        <Group justify="flex-end" mb="md">
          <Button onClick={() => setOpened(true)}>Upload Dataset</Button>
        </Group>
        <Grid>
          {datasets.map((d) => (
            <Grid.Col span={{ base: 12, sm: 6, md: 4 }} key={d}>
              <DatasetCard
                name={d}
                onView={() => {}}
                onDelete={() => handleDelete(d)}
              />
            </Grid.Col>
          ))}
        </Grid>
      </main>
      <Modal opened={opened} onClose={() => setOpened(false)} title="Upload Dataset">
        <FileInput label="Dataset File" value={file} onChange={setFile} />
        <Button mt="md" onClick={handleUpload} disabled={!file}>
          Upload
        </Button>
      </Modal>
    </>
  );
}
