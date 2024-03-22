package com.example.boot_page.controller;

import com.example.boot_page.entity.Member;
import com.example.boot_page.repository.MemberRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;

@Controller
public class SignupController {

    private final MemberRepository memberRepository;

    @Autowired
    public SignupController(MemberRepository memberRepository) {
        this.memberRepository = memberRepository;
    }

    @GetMapping("/signup")
    public String signupForm() {
        return "signup";
    }

    @PostMapping("/signup")
    @Transactional
    public String signup(Member member) {
        System.out.println("Signup: " + member); // 로그 출력
        memberRepository.save(member);
        return "redirect:/login"; // 회원 가입 후 로그인 페이지로 리디렉션
    }
}
