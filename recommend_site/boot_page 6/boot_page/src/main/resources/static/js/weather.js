// weather.js
const weatherContainer = document.getElementById("weather");
const weatherTitle = document.getElementById("weather-title");
const weatherInfo = document.getElementById("weather-info");
const weatherIcon = document.getElementById("weather-icon");
const weatherTemperature = document.getElementById("weather-temperature");

const API_KEY = "API_KEY";

var globalPlace = ""

function getCurrentLocation(successCallback, errorCallback) {
    const options = {
        enableHighAccuracy: true,
        timeout: 10000,
        maximumAge: 0
    };
    navigator.geolocation.getCurrentPosition(successCallback, errorCallback, options);
}

function translateWeatherState(weatherState) {
    const weatherStateMap = {
        "Clear": "맑음",
        "Clouds": "구름 많음",
        "Rain": "비",
        "Snow": "눈",
    };
    return weatherStateMap[weatherState] || weatherState;
}

function getWeatherIconClass(weatherState) {
    const weatherIconMap = {
        "Clear": "fa-sun",
        "Clouds": "fa-cloud",
        "Rain": "fa-cloud-showers-heavy",
        "Snow": "fa-snowflake",
    };
    return weatherIconMap[weatherState] || "fa-cloud";
}

function getWeather(lat, lng) {
    fetch(`https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lng}&appid=${API_KEY}&units=metric`)
    .then(function(response) {
        return response.json();
    }).then(function(json) {
        const temperature = json.main.temp;
        const place = json.name;
        globalPlace = place;
        const weatherState = json.weather[0].main;
        const weatherStateKorean = translateWeatherState(weatherState);
        const iconClass = getWeatherIconClass(weatherState);
        weatherTitle.innerText = place;
        weatherInfo.innerText = weatherStateKorean;
        weatherTemperature.innerText = `${temperature}°C`;
        weatherIcon.className = '';
        weatherIcon.classList.add('fa-solid', iconClass);
                searchRestaurantsByCategory('한식');
                searchRestaurantsByCategory('양식');
                searchRestaurantsByCategory('일식');
                searchRestaurantsByCategory('중식');
    });
}

function handleGeoSuccess(position) {
    const latitude = position.coords.latitude;
    const longitude = position.coords.longitude;
    getWeather(latitude, longitude);
}

function handleGeoError() {
    console.log("위치 정보를 가져올 수 없습니다.");
}

function initWeather() {
    getCurrentLocation(handleGeoSuccess, handleGeoError);
}

initWeather();


