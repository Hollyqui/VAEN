import React from 'react';
import './App.scss';
import Content from './components/Content';
import Menu from './components/Menu';

function App() {
  const mockupNetwork = [
    {
    name: "hey",
    trainable: true,
    dtype: "real",
    id: 1,
    avg_weight: "G",
    avg_abs_weight: "string"
  },
  {
    name: "hey",
    trainable: true,
    dtype: "real",
    id: 1,
    avg_weight: "G",
    avg_abs_weight: "string"
  }
]


  return (
    <div className="App">
      <header className="App-header">
        <Menu />

        <Content 
          networkOrigin={mockupNetwork}
        />
      </header>
    </div>
  );
}

export default App;
