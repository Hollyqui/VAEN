import React from 'react';
import { BrowserRouter as Router, Switch } from 'react-router-dom';
import './App.scss';
import Content from './components/Content';
import Menu from './components/Menu';

function App() {

  return (
    <div className="App">
      <header className="App-header">
        <Router>

          <Menu />

          <Switch>
            <Content />
          </Switch>

        </Router>
      </header>
    </div>
  );
}

export default App;
