import React from 'react'
import MenuTab from './MenuTab'

function Menu(){
    return (
        <div className="menu-div">
            <MenuTab name={'Home'} route={'/'} />
            <MenuTab name={'Dataset'} route={'/dataset'} />
            <MenuTab name={'Network'} route={'/network'} />
            <MenuTab name={'Training'} route={'/training'} />
            <MenuTab name={'Training Metrics'} route={'/training-metrics'} />
            <MenuTab name={'Result Metrics'} route={'/result-metrics'} />
        </div>
    )
}

export default Menu;