import React from 'react';

function InspectedLayer(){
    console.log('inspecting')

    return(
        <div className="inspectedLayer-div">
            {/* hello, I'm getting inspected, {props.test} */}
        </div>
    )
}

export default InspectedLayer;