import React, { useState } from 'react'
import InspectedLayer from './InspectedLayer'

interface LayerStructure{
    name: string,
    trainable: boolean,
    dtype: string,
    id: number,
    avg_weight: string,
    avg_abs_weight: string
}

function Layer(props:{layer:LayerStructure}){
    const [inspected, setInspected] = useState(<div></div>)

    function layout(layer:LayerStructure) {
        let layoutPopulated: any[] = []
        
        const arrayedLayer = Object.entries(layer)

        arrayedLayer.forEach( e => {
            layoutPopulated.push( <p> {e[0]}:{e[1]} </p>)
        })

        return layoutPopulated
    }
    
    function inspectLayer(){
        setInspected(<InspectedLayer />)
        
    }

    return(
        <div className="layer-div" id={`layer-${props.layer.id}`} onClick={inspectLayer}>
            {layout(props.layer)}
            {inspected}
        </div>
    )
}

export default Layer;