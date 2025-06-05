"use client";
import { Card, Text, Group, Button } from '@mantine/core';

interface DatasetCardProps {
  name: string;
  onView: () => void;
  onDelete: () => void;
}

export default function DatasetCard({ name, onView, onDelete }: DatasetCardProps) {
  return (
    <Card shadow="sm" padding="lg" radius="md" withBorder>
      <Group justify="space-between" mb="sm">
        <Text fw={500}>{name}</Text>
      </Group>
      <Group justify="flex-end">
        <Button size="xs" variant="light" onClick={onView}>View</Button>
        <Button size="xs" color="red" onClick={onDelete}>Delete</Button>
      </Group>
    </Card>
  );
}
