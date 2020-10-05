import React from 'react';
import { BrowserRouter as Router, Switch } from 'react-router-dom';
import './App.scss';
import Content from './components/Content';
import Menu from './components/Menu';

function App() {
  const mockupNetwork = [
    {
    name: "hey",
    trainable: true,
    dtype: "real",
    id: 0,
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
        <Router>

          <Menu />

          <Switch>
            <Content 
              networkOrigin={mockupNetwork}
            />
          </Switch>

        </Router>
      </header>
    </div>
  );
}

export default App;
