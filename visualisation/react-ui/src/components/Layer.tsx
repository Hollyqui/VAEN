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
    const [stateInspected, setStateInspected] = useState(false)

    function layout(layer:LayerStructure) {
        let layoutPopulated: any[] = []
        
        const arrayedLayer = Object.entries(layer)

        arrayedLayer.forEach( e => {
            layoutPopulated.push( <p> {e[0]}:{e[1]} </p>)
        })

        return layoutPopulated
    }
    
    function inspectLayer(){
        if (!stateInspected) {
            setStateInspected(!stateInspected)
            setInspected(
                <InspectedLayer 
                    id={props.layer.id}
                />
            )

            document.getElementsByTagName('body')[0].addEventListener("click", handleRemovalOfInspectLayer
                // // console.log(e.target.className)
                // let target = e.target as HTMLElement
                // console.log(target.className)
                // if (target.className === 'inspectedLayer-div'){
                //     setInspected(<div></div>)
                //     setStateInspected(!stateInspected)
                //     document.getElementsByTagName('body')[0].removeEventListener('click', ())
                // }
            )
        } 
    }

    function handleRemovalOfInspectLayer(e:any){
        let target = e.target as HTMLElement
                console.log(target.className)
                if (target.id === `inspectedLayer-${props.layer.id}`){
                    setInspected(<div></div>)
                    setStateInspected(false)
                    document.getElementsByTagName('body')[0].removeEventListener('click', handleRemovalOfInspectLayer)
                }
    }
    

    return(
        <div className="layer-div" id={`layer-${props.layer.id}`} onClick={inspectLayer}>
            {layout(props.layer)}
            {inspected}
        </div>
    )
}

export default Layer;