// ttn-decode.js
// Jesse McDonald, 3-28-26
// JS code to upload to TTN to decode payload.

function decodeUplink(input) {
    const bytes = input.bytes;
  
    if (bytes.length !== 5) {
      return { errors: ["Expected 5 bytes"] };
    }
  
    const object_id = bytes[0] | (bytes[1] << 8);
    const vehicle_class = bytes[2];
    const truck_type = bytes[3];
    const count = bytes[4];
  
    return {
      data: {
        object_id: object_id,
        vehicle_class: vehicle_class,
        truck_type: truck_type,
        count: count
      }
    };
  }