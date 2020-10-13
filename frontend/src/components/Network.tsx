import React, { useState } from 'react'
import Layer from './Layer'
import Link from './Link'
import io from 'socket.io-client'

interface NetworkStructure{
    name: string,
    trainable: boolean,
    dtype: string,
    id: number,
    avg_weight: string,
    avg_abs_weight: string
    hidden?: boolean
}

function Network(props: {}) {
    let onLoad: NetworkStructure[] = [
        {
            name: '',
            trainable: false,
            dtype: '',
            id: 0,
            avg_weight: '',
            avg_abs_weight: '',
            hidden: true
        }
    ]
    const [network, setNetwork] = useState(onLoad)

    let socket = io('http://localhost:6969/network_data')
    socket.connect()

    function populateContents(ntw:NetworkStructure[]) {
        let contentsPopulated:any[] = []

        ntw.forEach(layer => {
            if (layer.hidden) { } else {
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
            }


        })

        return contentsPopulated
    }

    return(
        <div className="network-div">
            {
                populateContents(network)
            }
            
        </div>
    )
}

export default Network