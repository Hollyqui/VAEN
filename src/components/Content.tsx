import React, { useState } from 'react';
import Layer from './Layer';
import Link from './Link';

interface NetworkStructure{
    name: string,
    trainable: boolean,
    dtype: string,
    id: number,
    avg_weight: string,
    avg_abs_weight: string
}

function Content(props:{networkOrigin:NetworkStructure[]}){
    // console.log(networkOrigin)
    // networkOrigin = Object.values(networkOrigin)
    const [network, setNetwork] = useState(props.networkOrigin)

    function populateContents(ntw:NetworkStructure[]) {
        let contentsPopulated:any[] = []

        ntw.forEach(layer => {

            if (layer.avg_weight === undefined) {
                contentsPopulated.push( 
                    <Layer 
                        layer={layer}
                    />
                )
            } else {
                contentsPopulated.push( 
                    <Layer 
                        layer={layer}
                    />,
                    
                    <Link 
                        id={layer.id}
                        avg_weight={layer.avg_weight}
                        abs_weight={layer.avg_abs_weight}
                    />
                )
            }
        })

        return contentsPopulated
    }

    return(
        <div className="content-div">
            {
                populateContents(network)
            }
        </div>
    )
}

export default Content;