
let autocomplete;
function initAutocomplete(){
  autocomplete = new google.maps.places.Autocomplete(
    document.getElementById('autocomplete'),
    {
      componentRestrictions: {'country':['US']},
      fields:['adr_address','geometry']
    });

  autocomplete.addListener('place_changed', onPlaceChanged);
}

function onPlaceChanged(){
  var place = autocomplete.getPlace();

  if (!place.geometry){
    document.getElementById('autocomplete').placeholder = "Enter a valid address"
  }else{
    //document.getElementById('destination').innerHTML = place.name
  }
}

function convertTZ(date, tzString) {
    return new Date((typeof date === "string" ? new Date(date) : date).toLocaleString("en-US", {timeZone: tzString}));   
}
  
function setup(){
    const today = new Date();
    //const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
    const timezone = "America/Chicago"
    console.log("Timezone: " , timezone); 
    today_with_timezone = convertTZ(today,timezone);

    hour = today_with_timezone.getHours();
    if (hour < 10 ) {
        hour = "0" + hour;
    }

    minute = today_with_timezone.getMinutes();
    if (minute < 10 ) {
        minute = "0" + minute;
    }

    current_time = hour + ":" + minute;

    var year = today_with_timezone.toLocaleString("default", { year: "numeric" });
    var month = today_with_timezone.toLocaleString("default", { month: "2-digit" });
    var day = today_with_timezone.toLocaleString("default", { day: "2-digit" });
    var formattedDate = year + "-" + month + "-" + day;

    document.getElementById('input_date').value = formattedDate;
    document.getElementById('input_time').value = current_time;
    document.getElementById('input_date').min = formattedDate;

    document.getElementById('output_graph').style.display = "hidden";
   

  }

function gohome(){
  document.location.href="/";
}