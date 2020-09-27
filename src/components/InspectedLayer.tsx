import React from 'react';

function InspectedLayer(props:{id:number}){
    console.log('inspecting')

    return(
        <div className="inspectedLayer-div" id={`inspectedLayer-${props.id}`}>
            {/* hello, I'm getting inspected, {props.test} */}
        </div>
    )
}

export default InspectedLayer;