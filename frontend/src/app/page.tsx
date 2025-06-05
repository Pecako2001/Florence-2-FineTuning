"use client";
import Navbar from '../components/Navbar';

export default function Home() {
  return (
    <>
      <Navbar />
      <main className="p-8 text-center">
        <h1 className="text-3xl font-bold mb-4">AI Training Platform</h1>
        <p>Use the navigation above to manage datasets, training and evaluation.</p>
      </main>
    </>
  );
}
