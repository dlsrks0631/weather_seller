package com.example.boot_page.controller;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestTemplate;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;

@RestController
public class RestaurantController {

    @GetMapping("/searchRestaurants")
    public ResponseEntity<String> searchRestaurants(@RequestParam String location, @RequestParam String category) {
        try {
            String query = location + category + "맛집";
            // String encodedQuery = URLEncoder.encode(query, StandardCharsets.UTF_8.toString());
            String url = "https://openapi.naver.com/v1/search/local.json?query=" + query+ " &display=" + 5 +"&sort="+"random";

            HttpHeaders headers = new HttpHeaders();
            headers.set("X-Naver-Client-Id", "x06K412MJqLTqjvNIfZg");
            headers.set("X-Naver-Client-Secret", "eL8TMO0vrM");
            HttpEntity<String> entity = new HttpEntity<>("", headers); // 비어 있는 Body와 함께 헤더를 전달

            RestTemplate restTemplate = new RestTemplate();
            ResponseEntity<String> response = restTemplate.exchange(url, HttpMethod.GET, entity, String.class);

            return response;
        } catch (Exception e) {
            e.printStackTrace();
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("Internal Server Error");
        }
    }

}
