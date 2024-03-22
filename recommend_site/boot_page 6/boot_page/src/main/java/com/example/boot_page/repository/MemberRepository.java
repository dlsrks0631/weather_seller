package com.example.boot_page.repository;

import com.example.boot_page.entity.Member;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface MemberRepository extends JpaRepository<Member, Long> {
    // 사용자 이름으로 Member 조회
    Member findByUsername(String username);

    // 사용자 이름과 비밀번호로 Member 조회 (이미 존재하는 메소드)
    Member findByUsernameAndPassword(String username, String password);
}
