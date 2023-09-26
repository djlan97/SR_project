
"use strict";

let Analog = require('./Analog.js');
let RobotStateRTMsg = require('./RobotStateRTMsg.js');
let Digital = require('./Digital.js');
let MasterboardDataMsg = require('./MasterboardDataMsg.js');
let IOStates = require('./IOStates.js');
let RobotModeDataMsg = require('./RobotModeDataMsg.js');
let ToolDataMsg = require('./ToolDataMsg.js');

module.exports = {
  Analog: Analog,
  RobotStateRTMsg: RobotStateRTMsg,
  Digital: Digital,
  MasterboardDataMsg: MasterboardDataMsg,
  IOStates: IOStates,
  RobotModeDataMsg: RobotModeDataMsg,
  ToolDataMsg: ToolDataMsg,
};
