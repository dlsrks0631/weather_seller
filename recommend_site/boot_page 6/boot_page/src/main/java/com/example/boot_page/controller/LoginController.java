package com.example.boot_page.controller;

import com.example.boot_page.entity.Member;
import com.example.boot_page.repository.MemberRepository;
import org.json.JSONException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;

import org.springframework.web.servlet.mvc.support.RedirectAttributes;
import org.springframework.http.*;
import org.springframework.web.client.RestTemplate;

import java.util.*;
import java.util.Collections;
import java.util.Iterator;
import java.util.Map;
import java.util.HashMap;

import org.json.JSONObject;

@Controller
public class LoginController {
    private final MemberRepository memberRepository;
    private final RestTemplate restTemplate;

    @Autowired
    public LoginController(MemberRepository memberRepository, RestTemplate restTemplate) {
        this.memberRepository = memberRepository;
        this.restTemplate = restTemplate;
    }

    @GetMapping("/login")
    public String loginForm() {
        return "login";
    }

    @PostMapping("/login")
    public String login(@RequestParam String username, @RequestParam String password, RedirectAttributes redirectAttributes) {
        Member member = memberRepository.findByUsernameAndPassword(username, password);
        if (member != null) {
            // 터미널에 username 출력
            System.out.println("Username: " + username);

            // API 요청을 보내고 결과를 처리
            String apiEndpoint = "http://localhost:8000/recommend";
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            headers.setAccept(Collections.singletonList(MediaType.APPLICATION_JSON));

            HttpEntity<String> request = new HttpEntity<>("{\"username\": \"" + username + "\"}", headers);

            ResponseEntity<String> response = restTemplate.postForEntity(apiEndpoint, request, String.class);

            //요청 받고 난 후 로직
            if (response.getStatusCode() == HttpStatus.OK) {
                String responseBody = response.getBody();
                // 받은 JSON 응답을 터미널에 출력
                System.out.println("Received JSON response: " + responseBody);

                // JSON 형식의 responseBody를 맵으로 변환
                Map<String, String> recommendations = parseJsonToMap(responseBody);

                // 변환된 맵 객체를 터미널에 출력
                System.out.println("Parsed recommendations: " + recommendations);

                // recommendations 맵을 RedirectAttributes에 추가
                redirectAttributes.addFlashAttribute("recommendations", recommendations);
            }

            redirectAttributes.addFlashAttribute("name", member.getName());
            return "redirect:/dashboard";
        } else {
            return "login";
        }
    }
    // JSON 문자열을 Map으로 변환하는 메서드
    private Map<String, String> parseJsonToMap(String json) {
        Map<String, String> unsortedMap = new HashMap<>();
        try {
            JSONObject jsonObject = new JSONObject(json);
            Iterator<String> keys = jsonObject.keys();

            while (keys.hasNext()) {
                String key = keys.next();
                String value = jsonObject.getString(key);
                unsortedMap.put(key, value);
            }

            // 백분율에 따라 Map을 정렬합니다.
            // Value(백분율)를 추출하여 Double로 변환하고, 내림차순으로 정렬합니다.
            List<Map.Entry<String, String>> list = new ArrayList<>(unsortedMap.entrySet());
            list.sort((entry1, entry2) -> {
                Double val1 = Double.valueOf(entry1.getValue().replace("%", ""));
                Double val2 = Double.valueOf(entry2.getValue().replace("%", ""));
                return val2.compareTo(val1);
            });

            // 정렬된 결과를 LinkedHashMap에 저장하여 순서를 유지합니다.
            Map<String, String> sortedMap = new LinkedHashMap<>();
            for (Map.Entry<String, String> entry : list) {
                sortedMap.put(entry.getKey(), entry.getValue());
            }

            return sortedMap;

        } catch (JSONException e) {
            e.printStackTrace();
            // JSON parsing error handling
        }
        // 오류가 발생한 경우, 비어 있는 Map을 반환합니다.
        return Collections.emptyMap();
    }
}