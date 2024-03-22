// restaurant.js
function searchRestaurantsByCategory(category) {
    $.ajax({
        method: "GET",
        url: "/searchRestaurants",
        data: {
            location: globalPlace,
            category: category
        }
    })
    .done(function(data) {
        if (category === '한식') {
            console.log(`Restaurant search response for ${category}: `, data);
            updateRestaurantListForMenu1(data.items, category);
        } else if (category === '양식') {
            console.log(`Restaurant search response for ${category}: `, data);
            updateRestaurantListForMenu2(data.items, category);
        } else if (category === '일식') {
            console.log(`Restaurant search response for ${category}: `, data);
            updateRestaurantListForMenu3(data.items, category)
        } else if (category === '중식') {
                    console.log(`Restaurant search response for ${category}: `, data);
                    updateRestaurantListForMenu4(data.items, category)
        }
    })
    .fail(function(error) {
        console.error("Error during restaurant search: ", error);
    });
}

// 한식
function updateRestaurantListForMenu1(items, category) {
    var restaurantElements = document.querySelectorAll(`.menu_display .restaurant-intro`);

    for (let i = 0; i < restaurantElements.length; i++) {
        if (i < items.length) {
            const item = items[i];
            const title = item.title.replace(/<[^>]+>/g, '');
            const link = item.link;
            restaurantElements[i].innerHTML = `
                <a href="${link}" target="_blank" style="text-decoration: none; color: black;"><h4 class="restaurant-name">${title}</h4></a>
                <p class="restaurant-location">위치: ${item.address}</p>
                <p class="restaurant-description">${item.description || '정보가 없습니다.'}</p>
            `;
        } else {
            restaurantElements[i].innerHTML = '<p>근처에 식당 정보가 없습니다.</p>';
        }
    }
}

// 양식
function updateRestaurantListForMenu2(items, category) {
    var restaurantElements = document.querySelectorAll(`.menu_display .restaurant-intro2`);

    for (let i = 0; i < restaurantElements.length; i++) {
        if (i < items.length) {
            const item = items[i];
            const title = item.title.replace(/<[^>]+>/g, '');
                        const link = item.link;
            restaurantElements[i].innerHTML = `
                <a href="${link}" target="_blank" style="text-decoration: none; color: black;"><h4 class="restaurant-name2">${title}</h4></a>
                <p class="restaurant-location2">위치: ${item.address}</p>
                <p class="restaurant-description2">${item.description || '정보가 없습니다.'}</p>
            `;
        } else {
            restaurantElements[i].innerHTML = '<p>근처에 식당 정보가 없습니다.</p>';
        }
    }
}

// 일식
function updateRestaurantListForMenu3(items, category) {
    var restaurantElements = document.querySelectorAll(`.menu_display .restaurant-intro3`);

    for (let i = 0; i < restaurantElements.length; i++) {
        if (i < items.length) {
            const item = items[i];
            const title = item.title.replace(/<[^>]+>/g, '');
                        const link = item.link;
            restaurantElements[i].innerHTML = `
                <a href="${link}" target="_blank" style="text-decoration: none; color: black;"><h4 class="restaurant-name3">${title}</h4></a>

                <p class="restaurant-location3">위치: ${item.address}</p>
                <p class="restaurant-description3">${item.description || '정보가 없습니다.'}</p>
            `;
        } else {
            restaurantElements[i].innerHTML = '<p>근처에 식당 정보가 없습니다.</p>';
        }
    }
}

// 중식
function updateRestaurantListForMenu4(items, category) {
    var restaurantElements = document.querySelectorAll(`.menu_display .restaurant-intro4`);

    for (let i = 0; i < restaurantElements.length; i++) {
        if (i < items.length) {
            const item = items[i];
            const title = item.title.replace(/<[^>]+>/g, '');
                        const link = item.link;
            restaurantElements[i].innerHTML = `
                <a href="${link}" target="_blank" style="text-decoration: none; color: black;"><h4 class="restaurant-name4">${title}</h4></a>

                <p class="restaurant-location4">위치: ${item.address}</p>
                <p class="restaurant-description4">${item.description || '정보가 없습니다.'}</p>
            `;
        } else {
            restaurantElements[i].innerHTML = '<p>근처에 식당 정보가 없습니다.</p>';
        }
    }
}

function initRestaurantSearch() {
    getCurrentLocation(function(position) {
        const lat = position.coords.latitude;
        const lng = position.coords.longitude;

//        searchRestaurantsByCategory('한식');
//        searchRestaurantsByCategory('양식');
//        searchRestaurantsByCategory('일식');
//        searchRestaurantsByCategory('중식');

    }, handleGeoError);
}

initRestaurantSearch();
