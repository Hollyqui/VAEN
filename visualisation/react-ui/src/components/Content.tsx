import React, { useState } from 'react';
import { Route } from 'react-router-dom';
import Dataset from './Dataset';
import Network from './Network';
import ResultMetrics from './ResultMetrics';
import Training from './Training';
import TrainingMetrics from './TrainingMetrics';
import Landing from './Landing';

interface NetworkStructure{
    name: string,
    trainable: boolean,
    dtype: string,
    id: number,
    avg_weight: string,
    avg_abs_weight: string
}

function Content(props: {networkOrigin: NetworkStructure[]}){

    return(
        <div className="content-div">
            <Route path="/">
                <Landing />
            </Route>

            
            <Route path="/dataset">
                <Dataset />
            </Route>

            <Route path="/network">
                <Network 
                    networkOrigin={props.networkOrigin}
                />
            </Route>

            <Route path="/training">
                <Training />
            </Route>

            <Route path="/training-metrics">
                <TrainingMetrics />
            </Route>

            <Route path="/result-metrics">
                <ResultMetrics />
            </Route>
        </div>
    )
}

export default Content;