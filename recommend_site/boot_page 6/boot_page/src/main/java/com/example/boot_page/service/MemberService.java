package com.example.boot_page.service;

import com.example.boot_page.repository.MemberRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import com.example.boot_page.entity.Member;

@Service
public class MemberService {

    private final MemberRepository memberRepository;

    @Autowired
    public MemberService(MemberRepository memberRepository) {
        this.memberRepository = memberRepository;
    }

    // 사용자 이름을 기반으로 Member 객체를 조회하는 메소드
    public Member getMemberByUsername(String username) {
        return memberRepository.findByUsername(username);
    }
}
