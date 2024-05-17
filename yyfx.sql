/*
 Navicat Premium Data Transfer

 Source Server         : Local Database
 Source Server Type    : MySQL
 Source Server Version : 80029 (8.0.29)
 Source Host           : localhost:3306
 Source Schema         : part2

 Target Server Type    : MySQL
 Target Server Version : 80029 (8.0.29)
 File Encoding         : 65001

 Date: 15/10/2024 12:28:02
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for Depression Analysis
-- ----------------------------
DROP TABLE IF EXISTS `Depression Analysis`;
CREATE TABLE `Depression Analysis`  (
  `question` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `answer` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of Depression Analysis
-- ----------------------------
INSERT INTO `Depression Analysis` VALUES ('Basic Information', 'Depression is a common mental illness characterized by low mood, reduced interest, pessimism, slow thinking, lack of initiative, self-blame, poor diet and sleep, worry about various illnesses, feeling of discomfort in many parts of the body, and in severe cases, suicidal thoughts and behaviors. Depression is the mental illness with the highest suicide rate. The incidence of depression is very high, with almost 1 in 7 adults being affected, making it akin to the \"common cold\" of psychiatry. It has become the second most burdensome disease worldwide, causing immense suffering for patients and their families, and significant losses to society. The main reason for this situation is the lack of proper understanding and the stigma surrounding depression, which discourages patients from seeking psychiatric treatment. In China, only 2% of depression patients have received treatment. Many patients do not get timely diagnosis and treatment, leading to worsening conditions and even severe consequences like suicide. Moreover, the lack of knowledge about depression leads to misinterpretation of symptoms as moodiness, failing to provide the necessary understanding and emotional support, thus exacerbating the condition.');
INSERT INTO `Depression Analysis` VALUES ('Types of Depression', '1. Endogenous Depression: characterized by five signs - laziness, dullness, change, worry, and anxiety (relative or absolute deficiency of brain biogenic amines).\n2. Hidden Depression: low mood and depressive symptoms are not obvious, often manifested as various physical discomforts such as palpitations, chest tightness, upper abdominal discomfort, shortness of breath, sweating, weight loss, insomnia, etc.\n3. Adolescent Depression: leads to learning difficulties, distracted attention, memory decline, overall or sudden decline in grades, school aversion, school phobia, or school refusal.\n4. Secondary Depression: for example, patients with high blood pressure, after taking antihypertensive drugs, leads to persistent depression and melancholy.\n5. Postpartum Depression: characterized by strong feelings of guilt, inferiority, hatred, fear, or aversion to one\'s own baby. Crying, insomnia, loss of appetite, and depression are common symptoms of this type of depression.\n6. White-collar Depression: young men and women with depression have disorders of the neuroendocrine system, and normal physiological cycles are also disrupted, with a variety of symptoms including mental depression, low mood, idleness, brooding, overthinking, insomnia, dreaminess, dizziness, forgetfulness, etc. In addition to major mental symptoms, there are also symptoms of digestive absorption dysfunction such as anorexia, nausea, vomiting, and bloating, and gynecological symptoms such as menstrual irregularities and dysmenorrhea.');
INSERT INTO `Depression Analysis` VALUES ('Core Symptoms of Depressive Episodes', '(1) Depressed mood, clearly abnormal for the individual, present for most of the day, almost every day, for at least 2 weeks; (2) Loss of interest or pleasure in activities that are usually enjoyable; (3) Fatigue or loss of energy.');
INSERT INTO `Depression Analysis` VALUES ('Additional Symptoms of Depressive Episodes', '(1) Loss of confidence and self-esteem; (2) Unreasonable self-reproach or excessive and inappropriate guilt; (3) Recurrent thoughts of death or suicide, or any suicidal behavior; (4) Evidence of decreased ability to think or concentrate, such as indecisiveness or hesitation; (5) Psychomotor changes, either agitation or retardation; (6) Any type of sleep disturbance; (7) Appetite changes (either decrease or increase) with corresponding weight changes.');
INSERT INTO `Depression Analysis` VALUES ('How to Help Yourself if You Have Depression', '1. Seek professional help: Consult with mental health professionals such as therapists, psychologists, or psychiatrists for professional support and treatment recommendations.\n2. Medication: In some cases, doctors may prescribe antidepressants, which should be used under professional supervision.\n3. Build a support system: Share your feelings with family and friends and seek their support. Having someone to talk to and provide emotional support can be helpful.\n4. Maintain a healthy lifestyle: Regular diet, adequate sleep, and moderate exercise positively impact mental health. Ensure your body is well cared for.\n5. Create a schedule: Establish a regular schedule, including fixed activities and social time, to help maintain structure and stability.\n6. Learn coping skills: Learn effective coping skills such as meditation, deep breathing, and progressive muscle relaxation to help alleviate anxiety and stress.\n7. Avoid isolation: Try to avoid isolating yourself. Engaging in social activities, even small interactions, can positively impact your mood.\n8. Accept yourself: Understand that depression is an illness, not a personal failure. Accept your feelings and seek support and treatment.');
INSERT INTO `Depression Analysis` VALUES ('Causes of Depression in the Elderly', 'Depression causes are more common among the elderly. For example, the elderly are more likely to experience loss, such as losing loved ones or familiar home environments. Other stressors that can trigger depression also increase, such as reduced income, worsening chronic diseases, and becoming isolated from former friends or social circles.');
INSERT INTO `Depression Analysis` VALUES ('Differences Between Depression and Dementia in the Elderly', 'For the elderly, depression can lead to symptoms similar to dementia: slowed thinking, reduced attention, confusion, and memory difficulties, not the sadness usually associated with depression. Doctors can distinguish between depression and dementia because, when depression is treated, patients return to a normal mental state. Dementia patients do not. Additionally, depressed patients may complain about memory loss but rarely forget important recent or personal events. In contrast, dementia patients often deny memory loss.');
INSERT INTO `Depression Analysis` VALUES ('Diagnosis of Depression in the Elderly', 'Diagnosing depression in the elderly is difficult for several reasons: \nDue to the reduced social activities or work, various depressive symptoms are hard to detect in time.\nSome elderly patients feel that depression is a weakness and are ashamed to tell others about their sadness or other symptoms.\nEmotional loss in the elderly is often... (content continues).');

SET FOREIGN_KEY_CHECKS = 1;
