package com.example.boot_page.controller;

import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.servlet.mvc.support.RedirectAttributes;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;

import java.util.Map;

@Controller
public class DashboardController {
    @GetMapping("/dashboard")
    public String dashboard(Model model, @ModelAttribute("name") String name, @ModelAttribute("recommendations") Map<String, String> recommendations) {
        System.out.println("Recommendations: " + recommendations);
        model.addAttribute("name", name);
        model.addAttribute("recommendations", recommendations);
        return "dashboard";
    }
}


